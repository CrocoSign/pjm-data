# pjm-data

Data preparation for CrocoSign, a Polish Sign Language (PJM) to text translation system built as an engineering thesis at PJATK by Wojciech Sieńko and Rostyslav Kovalenko. This repository turns the Open Repository of the PJM Corpus (Sign Language Linguistics Lab, University of Warsaw) into a machine learning dataset: one-sentence video clips with Polish and English text and a signer ID for every clip.

The result is published as the Hugging Face dataset [croco-corp/pjm-data](https://huggingface.co/datasets/croco-corp/pjm-data): 29,932 video-sentence pairs, about 25 hours, 150 Deaf signers. [https://crocosign.github.io](https://crocosign.github.io) is the overview of the whole project. The thesis itself is written in Polish.

The corpus is licensed for non-commercial use only. Anything derived from it, including this dataset, inherits that restriction.

## Where this fits

All repositories are under https://github.com/crocosign.

1. **CrocoSign** (this repository): data preparation
2. [SpaMo-PJM](https://github.com/crocosign/SpaMo-PJM) and [Uni-Sign-PJM](https://github.com/crocosign/Uni-Sign-PJM): model training
3. [pjm-models](https://github.com/crocosign/pjm-models): inference
4. [croco-sign-backend](https://github.com/crocosign/croco-sign-backend) and [croco-sign-frontend](https://github.com/crocosign/croco-sign-frontend): web application

Steps 03 and 05 are also published as the standalone repositories [pjm-segmentation](https://github.com/croco-corp/pjm-segmentation) and [crop-params](https://github.com/croco-corp/crop-params). This repository is the canonical copy.

## Published data

| Repository | Contents |
|---|---|
| dataset [croco-corp/pjm-data](https://huggingface.co/datasets/croco-corp/pjm-data) | `videos/` and `translations/` (the per-signer videos and subtitles from step 02), `dialogs_data.csv` (the step 02 metadata), `splits/` (`pjm_data.csv` and the `split_*` CSV files from step 08), `vectors/` (`texts.h5`, `texts_eng.h5`, `texts_multilang_pjm.h5` and visual feature files), `checkpoints/` |
| dataset [croco-corp/pjm-segments](https://huggingface.co/datasets/croco-corp/pjm-segments) | The WebDataset shards from step 03: the unfiltered segments, about 40,000 of them. Step 08 starts from 39,817 sentences |
| model [croco-corp/pjm-models](https://huggingface.co/croco-corp/pjm-models) | Trained models |

The segments repository is not the 29,932-pair dataset. The 29,932 pairs are the result of the filters in step 08: 39,817 sentences, 33,897 after the five selection filters, 29,932 after the quality filter. The split files in `pjm-data/splits/` list the keys that survive.

## The pipeline

The pipeline is `dataset-processing/01` to `08`. There is no single entry point: each step is run by hand, and several are notebooks written for Google Colab. The intermediate files were passed between us through a private Google Drive and are not published, so a new run starts at step 01 and regenerates them. Where a notebook used to download such a file, it now carries a comment saying what the file contained.

| Step | What it does | Input | Output |
|---|---|---|---|
| 01 `01_parse_pjm_pages.ipynb` | Scrapes the corpus pages | Corpus page IDs | CSV of video URL, subtitle URL, video type, age, sex, task ID |
| 02 `02_split_dialog_videos.py` | Splits two-person dialog videos and their subtitles into a left and a right signer | CSV from 01 (dialog rows) | Per-signer videos and VTT files, `output.csv` |
| 03 `03_video_segmentation/` | Cuts videos into one clip per subtitle cue | CSV with video and subtitle URLs | WebDataset shards on Hugging Face |
| 04 `04_post-segment-cleaning/` | Fixes transcription typos, trims video and subtitles to equal length | Folders of full-length `.vtt` and `.mp4` | Corrected `.vtt` files |
| 05 `05_crop_params/` | Finds a per-clip square crop around the signer | Shards from 03 | LMDB of crop boxes |
| 06 `06_speaker_id/` | Gives every clip a signer ID | Metadata CSV, videos | Metadata CSV with `speaker_id` |
| 07 `07_text_translation/` | Translates Polish text to English | HDF5 of Polish texts | HDF5 of English texts with a source flag |
| 08 `08_subset__selection/` | Normalises text, filters clips, builds train/val/test splits | Metadata CSV, English HDF5 | Split CSV files |

Clip IDs have the form `{video_id:03d}_{index:03d}_{L|R|M}` (left signer, right signer, or a single signer). The ID leads back to the source recording in the metadata CSV, and it is the join key between the shards, the crop LMDB and the text HDF5.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12 for the root project (`pyproject.toml`, `uv.lock`), which covers steps 01 to 04 and 08. Steps 05 and 07 have their own `pyproject.toml` and `uv.lock`. Step 06 is two Colab notebooks that install their own packages (`%pip install deepface`; OpenCV, scikit-learn, SciPy and Matplotlib are the Colab defaults), so it is not in any lock file.
- `ffmpeg` and `ffprobe`. Steps 02 and 03 encode with `h264_nvenc`, so they need an FFmpeg build with NVENC and an NVIDIA GPU.
- Hunspell with the Polish dictionary at `/usr/share/hunspell/pl_PL.dic` and `.aff` (Debian: `hunspell-pl`), for step 04.
- A `.env` file with `GEMINI_API_KEYS` (comma separated, rotated when a key hits its quota) for step 04, and `DeepL_API_Key` for step 07.
- A Hugging Face token with write access to your own dataset repositories for steps 02 and 03. The defaults point to the `croco-corp` repositories, which you cannot write to.
- A CUDA GPU for step 05 (default device `cuda:0`).

```bash
uv sync
```

## Running the steps

Run paths are relative to the repository root unless stated.

**01.** Open `dataset-processing/01_parse_pjm_pages.ipynb`. `parse_page(id)` fetches `https://www.korpuspjm.uw.edu.pl/videos/{id}` and extracts the video URL, subtitle URL, breadcrumb video type, sex, age and task number. `parse_pages(ids, file_path)` writes them to a CSV with a `parse_status` column. The cells after it clean the result: keep `parse_status == ok`, drop the column, save.

**02.** The input CSV needs the columns `video_id, video_url, subtitles_url, video_type, age, sex, task_id`, and every row must be a dialog.

```bash
uv run python dataset-processing/02_split_dialog_videos.py INPUT.csv \
  --output-dir ./output --max-workers 4 --max-size-gb 2.0 \
  --hf-repo-id YOUR_NAME/pjm_dialogs
```

The left and right halves are cropped from the video, and the audio is dropped. VTT cues are assigned to a side by their `align:` setting. A record whose cue count differs from its `align:` count fails and is written to the error log. Age (`"32-45, 32-45"`) and sex (`"MM"`) are split into one value per side. Results are uploaded in batches to the Hugging Face repository, with `output.csv` as `data.csv`. Failures go to `output/failed_records.csv`. The default `--hf-repo-id` is `croco-corp/pjm-data`, so pass your own repository to upload anywhere else.

**03.** The input CSV needs `video_id, video_url, subtitles_url, speaker_side` (`left`, `right`, anything else counts as a single signer). Run from `dataset-processing`:

```bash
cd dataset-processing
uv run python -m 03_video_segmentation.main --csv-path INPUT.csv \
  --hf-repo YOUR_NAME/pjm-segments --max-workers 4 --files-per-shard 500
```

Python prints a `RuntimeWarning` about the module name because the directory starts with a digit. It is harmless. Each video is re-encoded with a short GOP (`--gop`, default 2, no B-frames) so that FFmpeg can cut at nearly any cue boundary by stream copy. Each cue start is moved earlier by `--offset-millis` (default 100). Shards `train-NNNNNN.tar` hold `{key}.mp4` and `{key}.txt` and are uploaded as they fill. Failures are written to `logs/failed_records.csv` and `logs/failed_segments.csv`. `--start_shard` (with an underscore) only sets the number of the first shard. It does not skip records that were already processed.

**04.** Two scripts that work on the downloaded full-length videos and full VTT files. Their output does not feed step 03: the segmentation reads video and subtitle URLs from its input CSV, and nothing in this repository reads the cleaned files back. Each script is run from its own directory because the imports are not packaged. They read from `dataset-processing/data/`.

```bash
mkdir -p dataset-processing/data/vtts dataset-processing/data/videos
cd dataset-processing/04_post-segment-cleaning/transcriptions_pipeline
mkdir -p .logs
uv run python text_pipeline.py     # data/vtts -> data/vtts_fixed

cd ../matcher_pipeline
uv run python match_video_length_and_transcription.py
```

The text pipeline checks every word with a regex (letter repetitions, mixed case, all caps, longer than 20 characters) and then Hunspell. A word that fails goes to Gemini 2.5 Flash for a correction. A second Gemini call must accept the correction, otherwise it retries once and then keeps the original word. `BielikCorrector` (Bielik-11B-v2.6-Instruct, local GPU) is implemented in `correctors.py` but `main()` uses Gemini. It also writes `lemma_statistics.json` to the current directory. The matcher compares each video's duration (`ffprobe`) with the last cue end in the same-named `.vtt` and, if they differ by more than 1.0 s, cuts both to the last cue that fits. It overwrites the files in `data/videos` and `data/vtts` in place, so work on a copy. The text pipeline writes `data/vtts_fixed` while the matcher reads and edits `data/vtts`, so nothing joins the two outputs. We do not know from the code whether the corrected files were copied over `vtts` by hand before the matcher ran, or the matcher ran first.

**05.** This directory is self-contained (its own `src/crop` package, `pyproject.toml` and `uv.lock`); [crop-params](https://github.com/croco-corp/crop-params) is a copy of it:

```bash
cd dataset-processing/05_crop_params
uv sync
uv run python main.py --dataset-path YOUR_NAME/pjm-segments --output output
```

**06.** Both notebooks were run in Colab, and their outputs are cleared because they contained private Drive paths and links to a personal Hugging Face account. Their input files (the metadata CSV from steps 01 and 02, `speaker_id_singles.json` and the embedding pickles) were on a private Drive and are not published; each notebook's first cell says what the files contained. The result is published as `splits/pjm_data.csv` in croco-corp/pjm-data. The method:

- `speaker_id_age_sex.ipynb`: for single-signer videos, DeepFace VGG-Face embeddings of 5 evenly spaced frames are averaged per video, then clustered inside each age and sex group (agglomerative, cosine distance threshold 0.35). IDs look like `F_18-30_001`. The cells that export `speaker_groups.json` and the contact sheet for manual review are hard-coded to one group (`60+`, `M`) and must be edited per group.
- `speaker_id_dialogs.ipynb`: each dialog half is matched to the nearest speaker centroid among single-signer speakers with the same age and sex, then a hand-written `corrections` list overrides wrong matches, and age and sex are overwritten from the single-signer ground truth. After the automatic clustering and manual verification we had 150 distinct signers.

**07.** Needs Python 3.13. `data/texts.h5` is the Polish caption of every clip, one HDF5 dataset per clip key. No other step creates it, so `build_texts_h5.py` builds it from the `txt` field of the croco-corp/pjm-segments shards (unverified: we ran an equivalent script in the SpaMo-PJM repository, not this copy). The finished `texts.h5` and `texts_eng.h5` are also in `pjm-data/vectors/`.

```bash
cd dataset-processing/07_text_translation
uv sync
uv run --with datasets python build_texts_h5.py
uv run python main.py
```

`main.py` does not translate anything on its own. It expects `./data/texts.h5` and `./data/texts_eng.h5` to exist, then flags existing translations as `deepl`, applies the manual fixes in `problematic.json` (flag `manual`), and flags suspected untranslated lines as `invalid`. To translate, call `Translator(source_file, destination_file).translate()`, which sends batches of 50 sentences to DeepL (PL to EN-US) and resumes by skipping keys that already have a translation. Step 08 drops clips flagged `invalid`.

**08.** Notebooks with Polish prose. Run Jupyter from `dataset-processing/08_subset__selection`: they import the `hdf5_manager.py` next to them (an identical copy of the one in step 07) and read `./data/`. They need `data/texts_eng.h5` (from step 07 or `pjm-data/vectors/`); `01_analysis.ipynb` downloads `data/pjm_data.csv` from `pjm-data/splits/` if it is missing.

1. `01_analysis.ipynb`: lowercase, expand contractions, strip diacritics and punctuation, spaCy POS, NER and lemmas, mask rare words, write `data/processed_texts.csv`.
2. `02_selection.ipynb`: sequential keep or drop filters, writes `train_selection.csv`, `selection_analysis.csv` and the first `split_{train,val,test}.csv`. One analysis cell reads `data/crop_params.lmdb`, the output of step 05 (copy it to that path).
3. `03_data_quality_report.ipynb`: inspects problem samples and generates filtered splits.
4. `04_final_pipeline.ipynb`: the end-to-end version that produces the splits used for training: `data/split_{train,val,test}.csv`, `_filtered`, and `_ms`. The signer-independent split assigns whole signers to a set; the multi-signer split puts 80/10/10 of each signer's clips in train, val and test (seed 42).

## Known problems

- Step 04 writes corrected subtitles to `data/vtts_fixed` but the matcher reads and edits `data/vtts`. Nothing in the repository joins the two outputs.
- Free-tier Gemini limits are hard-coded in `gemini_rate_limit.py` (100 requests per day and 5 per minute per key for Flash), so step 04 needs several keys or a long run.
- Clip `.mp4` entries in the shards contain MPEG-TS bytes (`-f mpegts`), not an MP4 container. Step 05 relies on this and writes the bytes to a `.ts` temporary file.
- Steps 06 and 08 need inputs that are not published (see their sections), so they cannot be rerun end to end from this repository alone.

## License

Our code is under the MIT licence (see `LICENSE`).

The Open Repository of the PJM Corpus allows non-commercial use only.
