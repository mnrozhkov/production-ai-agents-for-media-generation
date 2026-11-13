# Capstone Proposal Template (One Page)

Copy this template to draft your capstone proposal at the end of Module 1. See the full [Capstone Project requirements](README.md) for the final submission checklist and peer-review rubric.

---

## Your Proposal

### 1. Problem and User
- **Who is the user or audience?**
- **What personalized media asset set does each run produce, and why does it matter?**

### 2. Input Records (`CSV` or `JSONL`)
- **Dataset source (synthetic or your own records):**
- **Schema (`id` column + required fields):**
- **Target batch size for the capstone:** 100 records for image/audio, 10 sampled records for video.

### 3. GPU Media Models (At Least Two Modalities)
| Role | Exact Model ID & Revision | Modality | Target GPU & Preset | Licence Link |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| Optional Model 3 | | | | |

### 4. Pipeline & Quality Checks
- **Brief → Planner (LLM) → GPU Tools → Judge/Checks → Object Storage:**
- **Which step runs asynchronously (Job or ticketed video generation)?**
- **Two observable acceptance criteria the judge/check will enforce before accepting an asset:**
  1.
  2.

### 5. Initial Budget Estimate (Using the Lesson 1.5 Cost Model)
- **GPU hourly rate (`$/h`) and source link:**
- **Marginal cost per asset (`c = rate × t_gen`) for each model:**
- **Fixed startup + load overhead (`F = rate × (t_start + t_ready)`) per worker run:**
- **Estimated total GPU cost for the capstone batch (100 image/speech records + 10 video records + ~20% retry allowance):**

---

## Filled Example: Personalized Holiday Storybook Factory

### 1. Problem and User
Parents and community organizers running a holiday gift drive want to send every child a personalized illustrated gift card, a warm spoken voice greeting, and a short animated keepsake clip based on their wishlist.

### 2. Input Records (`CSV` or `JSONL`)
- **Dataset:** `data/kids_100.csv` (synthetic child profiles, no real personal data).
- **Columns:** `id`, `name`, `age`, `wishlist`, `style`.
- **Batch size:** 100 records for `image` + `speech`, with a deterministic 10-record sample (`seed=42`) escalated to `video`.

### 3. GPU Media Models
| Role | Exact Model ID & Revision | Modality | Target GPU & Preset | Licence Link |
|---|---|---|---|---|
| `image` | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (`ac0da2ff...`) | Text-to-image (`1024x1024` PNG) | `gpu-l40s-a` (`1gpu-8vcpu-32gb`) | [Apache-2.0](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers/blob/main/LICENSE) |
| `speech` | `hexgrad/Kokoro-82M` (`f3ff3571...`) | Text-to-speech (`MP3` narration) | `gpu-l40s-a` (`1gpu-8vcpu-32gb`) | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M) |
| `video` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (`b8fff731...`) | Image/Text-to-video (`832x480` MP4, 5 s) | `gpu-l40s-a` (`1gpu-8vcpu-32gb`) | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

### 4. Pipeline & Quality Checks
1. **Plan (`llm`):** Generates structured JSON with card copy, `image_prompt`, `speech_text`, and `motion_prompt`.
2. **Keyframe (`image`) + Judge:** Renders the `1024x1024` illustration and checks two criteria: (a) the requested hero wishlist object is clearly visible, and (b) the image contains no readable text or watermarks. Failed keyframes retry up to 2 times.
3. **Narration (`speech`):** Synthesizes the voice greeting with Kokoro-82M once the image passes.
4. **Animation (`video`, async Job):** Runs asynchronously as a GPU Job on the 10 sampled records whose keyframe passed the judge.

### 5. Initial Budget Estimate (`gpu-l40s-a` at `$1.5484/h`, `rate = $0.0004301/s`)
Using the Module 1 L40S baseline measurements (`[measured: 2026-11-02, gpu-l40s-a, 1gpu-8vcpu-32gb, eu-north1]`):
- **100 images (+ 20% retries = 120 generations) across 2 shard Jobs:**
  - Fixed cost `2 × F_image = 2 × $0.1140 = $0.2280`
  - Marginal cost `120 × c_image = 120 × ($0.0004301 × 4.42 s) = $0.2281`
- **100 voice clips across 1 shard Job:**
  - Fixed cost `1 × F_speech = $0.1042`
  - Marginal cost `100 × c_speech = 100 × ($0.0004301 × 1.88 s) = $0.0809`
- **10 videos across 2 shard Jobs:**
  - Fixed cost `2 × F_video = 2 × ($0.0004301 × 722.20 s) = $0.6212`
  - Marginal cost `10 × c_video = 10 × ($0.0004301 × 157.49 s) = $0.6774`
- **Estimated total GPU compute for the full capstone batch:** `~$1.94` (well within a `$5.00` capstone run budget).
