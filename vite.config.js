<<<<<<< Updated upstream
import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';




export default defineConfig({
=======
const fs = require('fs');
const path = require('path');

const srcImage = "C:/Users/Lenovo/.gemini/antigravity-ide/brain/e4a7aa18-f74b-4da3-9b02-a7d681775d5b/.user_uploaded/media_1790702293025.jpg";
const targets = [
  path.resolve(__dirname, "hero_bg.jpg"),
  path.resolve(__dirname, "public", "hero_bg.jpg"),
  path.resolve(__dirname, "src", "hero_bg.jpg")
];

try {
  if (fs.existsSync(srcImage)) {
    const data = fs.readFileSync(srcImage);
    for (const target of targets) {
      const dir = path.dirname(target);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(target, data);
    }
    console.log("HERO IMAGE COPIED SUCCESSFULLY");
  }
} catch (e) {
  console.error("Image copy error:", e);
}

module.exports = {
>>>>>>> Stashed changes
  server: {
    fs: {
      strict: false
    }
<<<<<<< Updated upstream
  },
  plugins: [

    {
      name: 'copy-hero-bg-plugin',
      configureServer(server) {
        try {
          const uploadedSrc = 'C:/Users/Lenovo/.gemini/antigravity-ide/brain/d3969839-f759-48d7-a79c-7d62b85b7235/.user_uploaded/media_1790658574896.jpg';
          const heroBgDst = path.resolve(__dirname, 'hero_bg.jpg');
          if (fs.existsSync(uploadedSrc)) {
            fs.copyFileSync(uploadedSrc, heroBgDst);
            console.log('hero_bg.jpg copied to frontend directory');
          } else {
            console.warn('hero_bg source not found:', uploadedSrc);
          }
        } catch (e) {
          console.error('Failed to copy hero_bg.jpg:', e);
        }
      },
      buildStart() {
        try {
          const uploadedSrc = 'C:/Users/Lenovo/.gemini/antigravity-ide/brain/d3969839-f759-48d7-a79c-7d62b85b7235/.user_uploaded/media_1790658574896.jpg';
          const heroBgDst = path.resolve(__dirname, 'hero_bg.jpg');
          if (fs.existsSync(uploadedSrc)) {
            fs.copyFileSync(uploadedSrc, heroBgDst);
            console.log('hero_bg.jpg copied (build)');
          }
        } catch (e) {
          console.error('Failed to copy hero_bg.jpg during build:', e);
        }
      }
    },
    {
      name: 'copy-assets-plugin',
      configureServer(server) {
        try {
          const brainDir = 'C:/Users/Lenovo/.gemini/antigravity-ide/brain/c48b87b0-3916-4bbc-8191-47cc94091be9';
          const cleanBgSrc = path.join(brainDir, 'earth_clean_bg_1790626515095.jpg');
          const cleanBgDst = path.resolve(__dirname, 'earth_clean_bg.jpg');
          if (fs.existsSync(cleanBgSrc)) {
            fs.copyFileSync(cleanBgSrc, cleanBgDst);
            console.log('SUCCESSFULLY COPIED earth_clean_bg.jpg to frontend directory');
          }
        } catch (e) {
          console.error('Failed to copy assets:', e);
        }
      }
    },
    {
      name: 'save-satellite-plugin',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url === '/api/save-transparent-satellite' && req.method === 'POST') {
            let body = '';
            req.on('data', chunk => {
              body += chunk.toString();
            });
            req.on('end', () => {
              try {
                const base64Data = body.replace(/^data:image\/png;base64,/, '');
                const buffer = Buffer.from(base64Data, 'base64');
                const outPath = path.resolve(__dirname, 'satellite_transparent.png');
                fs.writeFileSync(outPath, buffer);
                console.log('SUCCESSFULLY SAVED SATELLITE PNG:', outPath, buffer.length, 'bytes');
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true, size: buffer.length }));
              } catch (err) {
                console.error('ERROR SAVING SATELLITE PNG:', err);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
              }
            });
            return;
          }
          next();
        });
      }
    }
  ]
});

=======
  }
};
>>>>>>> Stashed changes
