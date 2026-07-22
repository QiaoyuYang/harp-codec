// Populate the audio-samples section from docs/audio/manifest.json.
// The manifest is written by scripts/build_demo.py.
//
// manifest.json format:
// {
//   "tiers": [ { "key": "orig", "label": "Original" }, ... ],
//   "samples": [ { "name": "...", "dir": "sample1", "files": { "orig": "original.wav", ... } } ]
// }

async function loadSamples() {
  const container = document.getElementById("samples");
  let manifest;
  try {
    const res = await fetch("audio/manifest.json", { cache: "no-store" });
    if (!res.ok) return; // keep the default "no samples" message
    manifest = await res.json();
  } catch (e) {
    return;
  }

  if (!manifest.samples || manifest.samples.length === 0) return;
  container.innerHTML = "";

  for (const sample of manifest.samples) {
    const card = document.createElement("div");
    card.className = "sample";

    const title = document.createElement("h3");
    title.textContent = sample.name;
    card.appendChild(title);

    const tracks = document.createElement("div");
    tracks.className = "tracks";

    manifest.tiers.forEach((tier, i) => {
      const file = sample.files[tier.key];
      if (!file) return;

      const track = document.createElement("div");
      // First tier (Reference / Original) spans the full width; the rest pair up.
      track.className = i === 0 ? "track full" : "track";

      const label = document.createElement("span");
      label.className = "label";
      label.textContent = tier.label;

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `audio/${sample.dir}/${file}`;

      track.appendChild(label);
      track.appendChild(audio);
      tracks.appendChild(track);
    });

    card.appendChild(tracks);
    container.appendChild(card);
  }
}

loadSamples();
