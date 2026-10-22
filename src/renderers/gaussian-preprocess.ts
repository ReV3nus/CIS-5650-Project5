import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import { c_histogram_block_rows, C, SortStuff } from '../sort/sort';

export interface GaussianSettings {
  gaussian_multiplier: number,
  use_f16: boolean,
  workgroup_size: number,
  frustum_culling: boolean,
}

export type PreprocessEntry = 'preprocess' | 'preprocess_tiles';

const c_size_splat = 32;      // struct Splat in preprocess.wgsl
const c_max_workgroups = 65535;

const prelude_f32 = `
alias real = f32;
fn unpack_real(v: u32) -> vec2<real> { return unpack2x16float(v); }
`;
const prelude_f16 = `
enable f16;
alias real = f16;
fn unpack_real(v: u32) -> vec2<real> { return bitcast<vec2<f16>>(v); }
`;

export function supports_f16(device: GPUDevice) {
  return device.features.has('shader-f16');
}

export class GaussianPreprocessor {
  readonly splat_buffer: GPUBuffer;
  readonly settings_buffer: GPUBuffer;
  readonly tile_params_buffer: GPUBuffer;

  private readonly pipeline_layout: GPUPipelineLayout;
  private readonly bind_groups: GPUBindGroup[];
  private readonly modules = new Map<boolean, GPUShaderModule>();
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private settings: GaussianSettings = {
    gaussian_multiplier: 1, use_f16: false, workgroup_size: 256, frustum_culling: true,
  };

  constructor(
    private readonly device: GPUDevice,
    private readonly pc: PointCloud,
    camera_buffer: GPUBuffer,
    sorter: SortStuff,
  ) {
    this.splat_buffer = device.createBuffer({
      label: 'splats',
      size: pc.num_points * c_size_splat,
      usage: GPUBufferUsage.STORAGE,
    });
    this.settings_buffer = device.createBuffer({
      label: 'render settings',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.tile_params_buffer = device.createBuffer({
      label: 'tile params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // explicit layouts so every pipeline variant can share the bind groups
    const uniform = (binding: number): GPUBindGroupLayoutEntry =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
    const storage = (binding: number, read_only = false): GPUBindGroupLayoutEntry =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: read_only ? 'read-only-storage' : 'storage' } });

    const layouts = [
      device.createBindGroupLayout({ label: 'preprocess uniforms', entries: [uniform(0), uniform(1), uniform(2)] }),
      device.createBindGroupLayout({ label: 'preprocess gaussians', entries: [storage(0, true), storage(1, true), storage(2)] }),
      device.createBindGroupLayout({ label: 'preprocess sort', entries: [storage(0), storage(1), storage(2), storage(3)] }),
    ];
    this.pipeline_layout = device.createPipelineLayout({ label: 'preprocess', bindGroupLayouts: layouts });

    this.bind_groups = [
      device.createBindGroup({
        label: 'preprocess uniforms',
        layout: layouts[0],
        entries: [
          { binding: 0, resource: { buffer: camera_buffer } },
          { binding: 1, resource: { buffer: this.settings_buffer } },
          { binding: 2, resource: { buffer: this.tile_params_buffer } },
        ],
      }),
      device.createBindGroup({
        label: 'preprocess gaussians',
        layout: layouts[1],
        entries: [
          { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
          { binding: 1, resource: { buffer: pc.sh_buffer } },
          { binding: 2, resource: { buffer: this.splat_buffer } },
        ],
      }),
      device.createBindGroup({
        label: 'preprocess sort',
        layout: layouts[2],
        entries: [
          { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
          { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
          { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
          { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        ],
      }),
    ];

    this.set_settings(this.settings);
  }

  destroy() {
    this.splat_buffer.destroy();
    this.settings_buffer.destroy();
    this.tile_params_buffer.destroy();
  }

  set_settings(settings: GaussianSettings) {
    this.settings = { ...settings, use_f16: settings.use_f16 && supports_f16(this.device) };
    const data = new ArrayBuffer(16);
    const f = new Float32Array(data);
    const u = new Uint32Array(data);
    f[0] = settings.gaussian_multiplier;
    f[1] = this.pc.sh_deg;
    u[2] = settings.frustum_culling ? 1 : 0;
    this.device.queue.writeBuffer(this.settings_buffer, 0, data);
  }

  private get_pipeline(entry: PreprocessEntry): GPUComputePipeline {
    const { use_f16, workgroup_size } = this.settings;
    const key = `${entry}-${use_f16}-${workgroup_size}`;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      let module = this.modules.get(use_f16);
      if (!module) {
        module = this.device.createShaderModule({
          label: use_f16 ? 'preprocess f16' : 'preprocess f32',
          code: (use_f16 ? prelude_f16 : prelude_f32) + preprocessWGSL,
        });
        this.modules.set(use_f16, module);
      }
      pipeline = this.device.createComputePipeline({
        label: key,
        layout: this.pipeline_layout,
        compute: {
          module,
          entryPoint: entry,
          constants: {
            workgroupSize: C.histogram_wg_size,
            sortKeyPerThread: c_histogram_block_rows,
            preprocessWorkgroupSize: workgroup_size,
          },
        },
      });
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  encode(encoder: GPUCommandEncoder, entry: PreprocessEntry, timestampWrites?: GPUComputePassTimestampWrites) {
    const groups = Math.ceil(this.pc.num_points / this.settings.workgroup_size);
    const x = Math.min(groups, c_max_workgroups);
    const y = Math.ceil(groups / x);

    const pass = encoder.beginComputePass({ label: entry, timestampWrites });
    pass.setPipeline(this.get_pipeline(entry));
    this.bind_groups.forEach((g, i) => pass.setBindGroup(i, g));
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }
}
