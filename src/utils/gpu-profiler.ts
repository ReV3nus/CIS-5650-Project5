// Async readback of pass timestamps plus one u32 counter. Frames are skipped while a map is pending.
export class GpuProfiler {
  private has_timestamps: boolean;
  private query_set?: GPUQuerySet;
  private resolve_buffer?: GPUBuffer;
  private read_buffer: GPUBuffer;
  private counter_offset: number;
  private recorded = false;
  private pending = false;

  // ms between consecutive timestamps
  durations: number[] = [];
  counter = 0;

  constructor(private device: GPUDevice, private timestamp_count: number) {
    this.has_timestamps = device.features.has('timestamp-query');
    this.counter_offset = this.has_timestamps ? timestamp_count * 8 : 0;
    if (this.has_timestamps) {
      this.query_set = device.createQuerySet({ type: 'timestamp', count: timestamp_count });
      this.resolve_buffer = device.createBuffer({
        label: 'timestamp resolve',
        size: timestamp_count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
    this.read_buffer = device.createBuffer({
      label: 'profiler readback',
      size: this.counter_offset + 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  writes(begin: number, end: number): GPUComputePassTimestampWrites | undefined {
    if (!this.query_set) return undefined;
    return { querySet: this.query_set, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end };
  }

  // last thing in the frame's encoder
  resolve(encoder: GPUCommandEncoder, counter_buffer: GPUBuffer) {
    if (this.pending) return;
    if (this.query_set) {
      encoder.resolveQuerySet(this.query_set, 0, this.timestamp_count, this.resolve_buffer!, 0);
      encoder.copyBufferToBuffer(this.resolve_buffer!, 0, this.read_buffer, 0, this.timestamp_count * 8);
    }
    encoder.copyBufferToBuffer(counter_buffer, 0, this.read_buffer, this.counter_offset, 4);
    this.recorded = true;
  }

  // after queue.submit
  after_submit() {
    if (!this.recorded || this.pending) return;
    this.recorded = false;
    this.pending = true;
    this.read_buffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = this.read_buffer.getMappedRange();
      if (this.has_timestamps) {
        const t = new BigInt64Array(data, 0, this.timestamp_count);
        const durations = [];
        for (let i = 1; i < this.timestamp_count; i++) {
          durations.push(Number(t[i] - t[i - 1]) / 1e6);
        }
        this.durations = durations;
      }
      this.counter = new Uint32Array(data, this.counter_offset, 1)[0];
      this.read_buffer.unmap();
      this.pending = false;
    }).catch(() => { this.pending = false; });
  }
}
