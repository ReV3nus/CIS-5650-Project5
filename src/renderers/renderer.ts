import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { default as get_renderer_gaussian_tile } from './gaussian-tile-renderer';
import { GaussianSettings, supports_f16 } from './gaussian-preprocess';
import { PointCloud } from '../utils/load';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
  after_submit?: () => void,
  stats?: () => { preprocess_ms: number, sort_ms: number, render_ms: number, visible: number },
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer, gaussian_tile?: GaussianRenderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let point_cloud: PointCloud | undefined;
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
    use_f16: false,
    workgroup_size: 256,
    frustum_culling: true,
    preprocess_ms: 0,
    sort_ms: 0,
    render_ms: 0,
    visible: 0,
    bench_frame_ms: 0,
  };
  let benchmarking = false;
  const f16_supported = supports_f16(device);

  const gaussian_settings = (): GaussianSettings => ({
    gaussian_multiplier: params.gaussian_multiplier,
    use_f16: params.use_f16,
    workgroup_size: params.workgroup_size,
    frustum_culling: params.frustum_culling,
  });
  const apply_settings = () => {
    const s = gaussian_settings();
    gaussian_renderer?.set_settings(s);
    renderers.gaussian_tile?.set_settings(s);
  };
  // tile renderer is memory hungry, create it lazily
  const select_renderer = (name: string) => {
    if (name === 'gaussian_tile' && !renderers.gaussian_tile && point_cloud) {
      renderers.gaussian_tile = get_renderer_gaussian_tile(point_cloud, device, presentation_format, camera.uniform_buffer, canvas);
      renderers.gaussian_tile.set_settings(gaussian_settings());
    }
    renderer = renderers[name];
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
        'gaussian (tile, compute)': 'gaussian_tile',
      }
    }).on('change', (e) => {
      select_renderer(e.value);
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const pc = await load(uploadedFile, device);
        point_cloud = pc;
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.set_settings(gaussian_settings());
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        select_renderer(params.renderer);
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      {min: 0, max: 1.5}
    ).on('change', () => {
      apply_settings();
    });
  }
  {
    pane.addInput(params, 'frustum_culling').on('change', apply_settings);
    pane.addInput(params, 'use_f16', { label: f16_supported ? 'use_f16' : 'use_f16 (n/a)' })
      .on('change', (e) => {
        if (e.value && !f16_supported) {
          console.warn('shader-f16 is not supported on this device');
        }
        apply_settings();
      });
    pane.addInput(params, 'workgroup_size', {
      options: { '32': 32, '64': 64, '128': 128, '256': 256 },
    }).on('change', apply_settings);
  }
  {
    const stats = pane.addFolder({ title: 'GPU stats', expanded: true });
    if (!device.features.has('timestamp-query')) {
      stats.addMonitor({ note: 'timestamp-query unsupported' }, 'note');
    }
    stats.addMonitor(params, 'preprocess_ms', { format: (v: number) => v.toFixed(3) });
    stats.addMonitor(params, 'sort_ms', { format: (v: number) => v.toFixed(3) });
    stats.addMonitor(params, 'render_ms', { format: (v: number) => v.toFixed(3) });
    stats.addMonitor(params, 'visible', { label: 'visible/keys', format: (v: number) => v.toFixed(0) });
    stats.addMonitor(params, 'bench_frame_ms', { label: 'bench frame ms', format: (v: number) => v.toFixed(3) });
    stats.addButton({ title: 'benchmark (100 frames)' }).on('click', () => { run_benchmark(100); });
  }

  // back-to-back frames without rAF/vsync; frame_ms is submit -> GPU done
  async function run_benchmark(frames = 100, warmup = 10) {
    if (!renderer || benchmarking) return undefined;
    benchmarking = true;
    const sum = { frame_ms: 0, preprocess_ms: 0, sort_ms: 0, render_ms: 0, visible: 0 };
    try {
      for (let i = 0; i < warmup + frames; i++) {
        const t0 = performance.now();
        const encoder = device.createCommandEncoder();
        renderer.frame(encoder, context.getCurrentTexture().createView());
        device.queue.submit([encoder.finish()]);
        renderer.after_submit?.();
        await device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        // give the readback a chance to resolve
        await new Promise((r) => setTimeout(r, 0));
        if (i >= warmup) {
          sum.frame_ms += t1 - t0;
          const s = renderer.stats?.();
          if (s) {
            sum.preprocess_ms += s.preprocess_ms;
            sum.sort_ms += s.sort_ms;
            sum.render_ms += s.render_ms;
            sum.visible += s.visible;
          }
        }
      }
    } finally {
      benchmarking = false;
    }
    const avg = Object.fromEntries(Object.entries(sum).map(([k, v]) => [k, v / frames]));
    params.bench_frame_ms = avg.frame_ms;
    console.log('benchmark', params.renderer, avg);
    return avg;
  }
  // for running from the console
  (window as any).run_benchmark = run_benchmark;

  document.addEventListener('keydown', (event) => {
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        if (cameras && i < cameras.length) {
          console.log(`set to camera preset ${i}`);
          camera.set_preset(cameras[i]);
        }
        break;
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded && !benchmarking) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
      renderer.after_submit?.();
      const stats = renderer.stats?.();
      if (stats) {
        Object.assign(params, stats);
      }
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
