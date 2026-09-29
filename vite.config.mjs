import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const srcImage = "C:/Users/Lenovo/.gemini/antigravity-ide/brain/e4a7aa18-f74b-4da3-9b02-a7d681775d5b/.user_uploaded/media_1790702293025.jpg";
const targets = [
  path.resolve(__dirname, "hero_bg.jpg"),
  path.resolve(__dirname, "public", "hero_bg.jpg"),
  path.resolve(__dirname, "src", "hero_bg.jpg"),
  path.resolve(__dirname, "1000121017.png")
];

try {
  if (fs.existsSync(srcImage)) {
    const data = fs.readFileSync(srcImage);
    for (const target of targets) {
      const dir = path.dirname(target);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(target, data);
    }
  }
} catch (e) {
  console.error("Image copy error:", e);
}

export default defineConfig({
  server: {
    fs: {
      strict: false
    }
  }
});
