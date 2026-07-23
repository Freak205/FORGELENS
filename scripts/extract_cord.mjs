/**
 * Extract CORD v2 images without native Parquet extensions.
 *
 * Windows Application Control blocks PyArrow's unsigned `_dataset` extension
 * on this machine. This script uses pinned MIT-licensed JavaScript/WASM readers
 * hosted entirely below F:\HYPERVERGE.
 */

import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const storageRoot = path.resolve(projectRoot, "..");
const revision = "7f0115a4b758a71d6473b8d085751692da2fef98";
const inputRoot = path.join(storageRoot, "data", "cord-v2", revision);
const outputRoot = path.join(storageRoot, "data", "cord-v2-extracted", revision);
const modulesRoot = path.join(storageRoot, "tools", "node", "node_modules");

const hyparquet = await import(
  pathToFileURL(path.join(modulesRoot, "hyparquet", "src", "node.js"))
);
const { compressors } = await import(
  pathToFileURL(path.join(modulesRoot, "hyparquet-compressors", "src", "index.js"))
);

const shards = [
  ["test", "test-00000-of-00001-9c204eb3f4e11791.parquet"],
  ["train", "train-00000-of-00004-b4aaeceff1d90ecb.parquet"],
  ["train", "train-00001-of-00004-7dbbe248962764c5.parquet"],
  ["train", "train-00002-of-00004-688fe1305a55e5cc.parquet"],
  ["train", "train-00003-of-00004-2d0cd200555ed7fd.parquet"],
  ["validation", "validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
];
const initializedSplits = new Set();

function extension(bytes) {
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e) return ".png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return ".jpg";
  throw new Error("unsupported embedded image format");
}

for (const [officialSplit, filename] of shards) {
  const splitRoot = path.join(outputRoot, officialSplit);
  const imageRoot = path.join(splitRoot, "images");
  await mkdir(imageRoot, { recursive: true });
  const manifestPath = path.join(splitRoot, "metadata.jsonl");
  const manifest = createWriteStream(manifestPath, {
    encoding: "utf8",
    flags: initializedSplits.has(officialSplit) ? "a" : "w",
  });
  initializedSplits.add(officialSplit);
  const file = await hyparquet.asyncBufferFromFile(
    path.join(inputRoot, "data", filename),
  );
  const rows = await hyparquet.parquetReadObjects({
    file,
    compressors,
    utf8: false,
  });
  let rowIndex = 0;
  for (const row of rows) {
    const groundTruth = JSON.parse(row.ground_truth);
    const imageId = String(groundTruth.meta?.image_id ?? rowIndex);
    const bytes = Buffer.from(row.image.bytes);
    const imageFilename = `${imageId.padStart(6, "0")}${extension(bytes)}`;
    await writeFile(path.join(imageRoot, imageFilename), bytes);
    const record = {
      sample_id: `cord-${officialSplit}-${imageId}`,
      source_group: `cord:${officialSplit}:${imageId}`,
      official_split: officialSplit,
      image_id: imageId,
      image_path: path.join("images", imageFilename).replaceAll("\\", "/"),
      sha256: createHash("sha256").update(bytes).digest("hex"),
      ground_truth: groundTruth,
    };
    if (!manifest.write(`${JSON.stringify(record)}\n`)) {
      await once(manifest, "drain");
    }
    rowIndex += 1;
  }
  manifest.end();
  await once(manifest, "finish");
  console.log(JSON.stringify({ split: officialSplit, rows: rowIndex }));
}
