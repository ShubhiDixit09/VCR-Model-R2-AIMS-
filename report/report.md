# Beyond the Image: Joint Answer and Rationale Selection for Visual Commonsense Reasoning

## 1. Problem and design goal

The task requires two constrained decisions for every image-question pair: select one of four answers, then select one of four supplied rationales that explains the answer. The system must therefore be visually grounded, preserve the dependency between answer and rationale, avoid free-text rationales, and run without human intervention.

The submitted system is an inference-only pipeline built around the frozen Qwen2.5-VL-3B-Instruct vision-language model. It uses 4-bit NF4 quantization so the complete experiment can run on a 16 GB Tesla T4. The central design choice is to treat VCR as constrained discrimination rather than ordinary text generation.

## 2. Architecture

VCR questions refer to detected objects using identifiers such as `[person0]`. A grounded visual view is constructed by drawing each supplied bounding box and its stable object identifier on the image. This makes the link between text references and image regions explicit.

For answer selection, the grounded image, question and four candidate answers are presented to the VLM. Instead of generating arbitrary text, the system reads the next-token logits only for labels A, B, C and D and normalises them into `P(A_i | I,Q)`.

For rationale selection, the process is repeated for every possible answer. Each prompt contains the image, question, a candidate answer and all four rationale choices. This produces a 4×4 matrix `P(R_j | I,Q,A_i)`. The final pair is selected from all sixteen combinations:

`(A*,R*) = argmax(i,j) P(A_i | I,Q) × P(R_j | I,Q,A_i)`.

This factorisation follows the task definition and prevents the rationale stage from being detached from its answer.

## 3. Experimental strategy

Development began with eight validation questions spanning three unique images. Images were cached once and then linked to all associated questions. Gold labels were stored separately and never inserted into model prompts.

Several controlled alternatives were evaluated. A full-sentence likelihood baseline was rejected because length and tokenisation effects made its option scores unstable. Constrained A–D scoring substantially improved pilot answer selection. Original, grounded and dual visual views were compared; grounding improved answer selection, while dual input did not help rationale selection. A 7B model matched or underperformed the 3B model on the pilot while using considerably more time and memory, so the 3B backbone was retained. Rationale permutation averaging reduced performance and was also rejected.

A separate 30-example benchmark was split into ten calibration and twenty evaluation examples. Temperature calibration selected 1.0 for answers and 1.5 for rationales on calibration, but its joint accuracy fell from 30% to 25% on evaluation. Because this did not generalise, the final system locked both temperatures at 1.0. A fresh, untouched twenty-example holdout was then used once for final reporting.

## 4. Results and analysis

On the locked twenty-example holdout, the system achieved 60% Q→A accuracy, 40% QA→R accuracy and 15% joint Q→AR accuracy. The first two values exceed their four-choice random baseline of 25%; the pair metric has a sixteen-pair random baseline of 6.25%.

The gap between answer and joint accuracy shows that selecting a visually plausible answer is easier than identifying the precise human-written explanation. Error inspection suggests three recurring difficulties: subtle social intent, rationales requiring context beyond visible objects, and several semantically plausible rationales with small score margins. The small holdout also gives high statistical uncertainty, so these figures should be treated as a diagnostic result rather than a benchmark claim.

Key design choices are therefore supported by ablations rather than selected after viewing the final holdout. The system is deterministic, reproducible, label-constrained and compliant with the no-human-inference requirement. Its main limitation is the absence of task-specific fine-tuning. A natural extension is LoRA training on the official training split followed by evaluation on a substantially larger image-disjoint set.
