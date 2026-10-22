import './style.css'
import init from './renderers/renderer';
import { assert } from './utils/util';

(async () => {
  
  if (navigator.gpu === undefined) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'WebGPU is not supported in this browser.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (adapter === null) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'No adapter is available for WebGPU.';
    return;
  }
  
  // optional, used for the f16 toggle and GPU timings
  const requiredFeatures: GPUFeatureName[] = [];
  for (const feature of ['shader-f16', 'timestamp-query'] as GPUFeatureName[]) {
    if (adapter.features.has(feature)) {
      requiredFeatures.push(feature);
    }
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: { 
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: adapter.limits. maxStorageBufferBindingSize
     },
  });

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  init(canvas, context, device);
  
})();