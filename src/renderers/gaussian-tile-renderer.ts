// Tile renderer: preprocess_tiles -> finalize -> sort -> identify_ranges -> rasterize -> blit.
// The key count is only known on the GPU; it's read back a frame or two later and the sort
// buffers grow if it didn't fit.
import { PointCloud } from '../utils/load';
import tileWGSL from '../shaders/tile_raster.wgsl';
import { get_sorter, c_histogram_block_rows, C, SortStuff } from '../sort/sort';
import { GaussianPreprocessor, GaussianSettings } from './gaussian-preprocess';
import { GpuProfiler } from '../utils/gpu-profiler';
import { GaussianRenderer } from './gaussian-renderer';

const c_tile_size = 16;
const c_initial_keys_per_gaussian = 4;
const c_bytes_per_key = 16;   // key + index, x2 for ping-pong
const c_max_sort_memory = 1536 * 1024 * 1024;

function max_capacity(device: GPUDevice) {
  const by_limit = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 4);
  // sorter rounds up to whole blocks plus one
  const block = C.histogram_wg_size * c_histogram_block_rows;
  return Math.min(by_limit - 2 * block, Math.floor(c_max_sort_memory / c_bytes_per_key));
}

function destroy_sorter(sorter: SortStuff) {
  sorter.sort_info_buffer.destroy();
  sorter.sort_dispatch_indirect_buffer.destroy();
  for (const p of sorter.ping_pong) {
    p.sort_depths_buffer.destroy();
    p.sort_indices_buffer.destroy();
  }
}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  canvas: HTMLCanvasElement,
): GaussianRenderer {

  const capacity_limit = max_capacity(device);

  const ranges_dispatch_buffer = device.createBuffer({
    label: 'ranges dispatch indirect',
    size: 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
  });
  const requested_keys_buffer = device.createBuffer({
    label: 'requested tile keys',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const module = device.createShaderModule({ label: 'tile raster', code: tileWGSL });
  const compute_pipeline = (entryPoint: string) => device.createComputePipeline({
    label: entryPoint,
    layout: 'auto',
    compute: {
      module,
      entryPoint,
      constants: entryPoint === 'finalize'
        ? { sortKeysPerDispatch: C.histogram_wg_size * c_histogram_block_rows }
        : {},
    },
  });
  const finalize_pipeline = compute_pipeline('finalize');
  const ranges_pipeline = compute_pipeline('identify_ranges');
  const raster_pipeline = compute_pipeline('rasterize');
  const blit_pipeline = device.createRenderPipeline({
    label: 'tile blit',
    layout: 'auto',
    vertex: { module, entryPoint: 'blit_vs' },
    fragment: { module, entryPoint: 'blit_fs', targets: [{ format: presentation_format }] },
    primitive: { topology: 'triangle-list' },
  });

  // depends on key capacity
  let settings: GaussianSettings | undefined;
  let capacity = 0;
  let sorter: SortStuff;
  let preprocessor: GaussianPreprocessor;
  let finalize_bind_group: GPUBindGroup;

  const allocate = (new_capacity: number) => {
    if (sorter) destroy_sorter(sorter);
    preprocessor?.destroy();

    capacity = Math.max(1, Math.min(new_capacity, capacity_limit));
    console.log(`tile renderer key capacity: ${capacity}`);
    sorter = get_sorter(capacity, device);
    preprocessor = new GaussianPreprocessor(device, pc, camera_buffer, sorter);
    if (settings) preprocessor.set_settings(settings);

    finalize_bind_group = device.createBindGroup({
      label: 'finalize',
      layout: finalize_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: preprocessor.tile_params_buffer } },
        { binding: 2, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        { binding: 4, resource: { buffer: ranges_dispatch_buffer } },
        { binding: 11, resource: { buffer: requested_keys_buffer } },
      ],
    });
    width = 0;   // rebuild bind groups below
  };

  // depends on canvas size
  let width = 0;
  let height = 0;
  let tiles_x = 0;
  let tiles_y = 0;
  let output_texture: GPUTexture | undefined;
  let ranges_buffer: GPUBuffer | undefined;
  let ranges_bind_group: GPUBindGroup;
  let raster_bind_group: GPUBindGroup;
  let blit_bind_group: GPUBindGroup;

  const resize = () => {
    width = Math.max(1, canvas.width);
    height = Math.max(1, canvas.height);
    tiles_x = Math.ceil(width / c_tile_size);
    tiles_y = Math.ceil(height / c_tile_size);
    const num_tiles = tiles_x * tiles_y;
    const tile_bits = Math.max(1, Math.ceil(Math.log2(num_tiles)));
    const depth_bits = 32 - tile_bits;

    device.queue.writeBuffer(preprocessor.tile_params_buffer, 0,
      new Uint32Array([tiles_x, tiles_y, depth_bits, capacity]));

    output_texture?.destroy();
    ranges_buffer?.destroy();
    output_texture = device.createTexture({
      label: 'tile raster output',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    ranges_buffer = device.createBuffer({
      label: 'tile ranges',
      size: num_tiles * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    ranges_bind_group = device.createBindGroup({
      label: 'identify ranges',
      layout: ranges_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: preprocessor.tile_params_buffer } },
        { binding: 2, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 5, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 6, resource: { buffer: ranges_buffer } },
      ],
    });
    raster_bind_group = device.createBindGroup({
      label: 'rasterize',
      layout: raster_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera_buffer } },
        { binding: 1, resource: { buffer: preprocessor.tile_params_buffer } },
        { binding: 6, resource: { buffer: ranges_buffer } },
        { binding: 7, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 8, resource: { buffer: preprocessor.splat_buffer } },
        { binding: 9, resource: output_texture.createView() },
      ],
    });
    blit_bind_group = device.createBindGroup({
      label: 'blit',
      layout: blit_pipeline.getBindGroupLayout(0),
      entries: [{ binding: 10, resource: output_texture.createView() }],
    });
  };

  allocate(pc.num_points * c_initial_keys_per_gaussian);

  // preprocess begin/end, raster begin/end
  const profiler = new GpuProfiler(device, 4);
  const nulling_data = new Uint32Array([0]);

  const frame = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const requested = profiler.counter;
    if (requested > capacity && capacity < capacity_limit) {
      allocate(Math.ceil(requested * 1.5));
    }
    if (canvas.width !== width || canvas.height !== height || !output_texture) {
      resize();
    }

    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    encoder.clearBuffer(ranges_buffer!);

    preprocessor.encode(encoder, 'preprocess_tiles', profiler.writes(0, 1));
    {
      const pass = encoder.beginComputePass({ label: 'finalize tile keys' });
      pass.setPipeline(finalize_pipeline);
      pass.setBindGroup(0, finalize_bind_group);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    sorter.sort(encoder);
    {
      const pass = encoder.beginComputePass({ label: 'identify tile ranges' });
      pass.setPipeline(ranges_pipeline);
      pass.setBindGroup(0, ranges_bind_group);
      pass.dispatchWorkgroupsIndirect(ranges_dispatch_buffer, 0);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'tile rasterize', timestampWrites: profiler.writes(2, 3) });
      pass.setPipeline(raster_pipeline);
      pass.setBindGroup(0, raster_bind_group);
      pass.dispatchWorkgroups(tiles_x, tiles_y);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        label: 'tile blit',
        colorAttachments: [{ view: texture_view, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(blit_pipeline);
      pass.setBindGroup(0, blit_bind_group);
      pass.draw(3);
      pass.end();
    }
    profiler.resolve(encoder, requested_keys_buffer);
  };

  return {
    frame,
    after_submit: () => profiler.after_submit(),
    stats: () => {
      const [preprocess_ms = 0, sort_ms = 0, render_ms = 0] = profiler.durations;
      // key count here, not gaussians
      return { preprocess_ms, sort_ms, render_ms, visible: profiler.counter };
    },
    set_settings: (s: GaussianSettings) => {
      settings = s;
      preprocessor.set_settings(s);
    },
    camera_buffer,
  };
}
