import { PointCloud } from '../utils/load';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter } from '../sort/sort';
import { Renderer } from './renderer';
import { GaussianPreprocessor, GaussianSettings } from './gaussian-preprocess';
import { GpuProfiler } from '../utils/gpu-profiler';

export interface GaussianRenderer extends Renderer {
  set_settings: (settings: GaussianSettings) => void,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // instance_count gets filled from sort_infos.keys_size each frame
  const indirect_draw_buffer = createBuffer(
    device,
    'gaussian indirect draw',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([4, 0, 0, 0]),
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocessor = new GaussianPreprocessor(device, pc, camera_buffer, sorter);

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ label: 'gaussian render', code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        // premultiplied, back to front
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian render',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: preprocessor.splat_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // preprocess begin/end, render begin/end
  const profiler = new GpuProfiler(device, 4);

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    // both are atomically incremented in the shader
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);
    preprocessor.encode(encoder, 'preprocess', profiler.writes(0, 1));
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);

    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [{
        view: texture_view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      timestampWrites: profiler.writes(2, 3),
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, render_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      preprocess(encoder);
      sorter.sort(encoder);
      render(encoder, texture_view);
      profiler.resolve(encoder, sorter.sort_info_buffer);
    },
    after_submit: () => profiler.after_submit(),
    stats: () => {
      const [preprocess_ms = 0, sort_ms = 0, render_ms = 0] = profiler.durations;
      return { preprocess_ms, sort_ms, render_ms, visible: profiler.counter };
    },
    set_settings: (settings: GaussianSettings) => preprocessor.set_settings(settings),
    camera_buffer,
  };
}
