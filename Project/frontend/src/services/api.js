import axios from 'axios';

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === 'string') {
        const comma = result.indexOf(',');
        resolve(comma >= 0 ? result.slice(comma + 1) : result);
      } else {
        reject(new Error('Failed to read file data.'));
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export async function uploadSketch(file, mode, settings = {}) {
  const base64Data = await readFileAsBase64(file);
  const response = await axios.post('/api/sketch', {
    mode,
    filename: file.name || 'sketch.png',
    data: base64Data,
    settings
  });
  return response.data;
}

export async function vectorizeRaster(file, strokeThickness = 0) {
  const base64Data = await readFileAsBase64(file);
  const response = await axios.post('/api/vectorize', {
    data: base64Data,
    strokeThickness
  });
  return response.data;
}

export async function fetchLatestRender() {
  const response = await axios.get('/api/result/latest');
  return response.data;
}
