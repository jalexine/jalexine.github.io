---
layout: post
title: "TILED ATTENTION: FLASHATTENTION 1 to 2"
date: 2026-01-20
category: technical
---

<span style="color:white">You don't really understand something until you play with it. Reimplementing FlashAttention pushed me to apply concepts from PMPP that otherwise stay abstract: like <span style="color:#ff2777">**memory hierarchy**</span>, <span style="color:#ff2777">**tiling**</span>, <span style="color:#ff2777">**synchronization**</span> and <span style="color:#ff2777">**kernel fusion**</span>.</span>



<span style="color:white">Here I walk through both <span style="color:#ff2777">**FlashAttention papers**</span> and explain the <span style="color:#ff2777">**key differences**</span> between them.</span>

<br>

## ATTENTION BRIEFLY

<span style="color:white">Before getting into FlashAttention itself, it's worth recalling briefly what attention actually computes. If you are not familiar with attention, I recommend reading <a href="https://arxiv.org/abs/1706.03762" style="color:#ff69b4">Attention Is All You Need</a> first.</span>

<br>

### >IN ONE ASCII

<span style="color:white">For a single query **q**, <span style="color:#ff2777">**attention**</span> compares it against a set of keys **K** and uses the result to combine corresponding values **V**.</span>

<br>

<pre style="color:#ff69b4">
           K0    K1    K2    ...    Kn    (keys)
            |     |     |           |
            v     v     v           v
q (query) -> q·K0  q·K1  q·K2   ...   q·Kn
                  [   softmax   ]
                         |
                         v
            w0*V0 + w1*V1 + w2*V2 + ... + wn*Vn   (values)
</pre>

<br>

<span style="color:white">Here <span style="color:#ff2777">**Q**</span> is the query, <span style="color:#ff2777">**K**</span> are the keys and <span style="color:#ff2777">**V**</span> are the values. Each query produces its own <span style="color:#ff2777">**softmax**</span> distribution over all keys. Softmax is applied per query.  So computing the output for a single query requires scanning all keys to compute dot products, tracking global statistics for <span style="color:#ff2777">**numerical stability**</span> and then scanning all values again to form the weighted sum.

<span style="color:white">On a GPU, the computation itself is fine. The real cost comes from memory traffic and synchronization.<span style="color:#ff2777">**K**</span> and <span style="color:#ff2777">**V**</span> are repeatedly from <span style="color:#ff2777">**global memory**</span> while global statistics (max and sum) must be maintained.</span>


<br>


### SLOW ON GPU

<span style="color:white">The problem is not compute, it’s memory.</span>

<span style="color:white">Standard attention writes the N×N attention matrix to HBM, reads it back for softmax, then reads it again to apply V. This moves a huge amount of data through slow memory.</span>

<span style="color:white">Softmax also requires global reductions, which prevents simple tiling and adds synchronization.
Then attention is memory-bound. FLOPs are cheap, memory traffic is not.</span>


<span style="color:white">So FlashAttention is designed to : <span style="color:#ff2777">**reduce memory movement**</span> while computing a stable softmax in a single <span style="color:#ff2777">**fused kernel**</span>.</span>

<br>

<br>

## FLASHATTENTION 1

<span style="color:white">The core idea is to never write the full N×N <span style="color:#ff2777">**attention matrix**</span>. Instead, compute it in small blocks that fit entirely in fast <span style="color:#ff2777">**on-chip memory**</span>.</span>

<span style="color:white">To understand why this matters, look at this beautiful figure i colored in pink and more specially at the memory hierarchy. 
<img src="../assets_lab/a1fig1.png" alt="flashattention architecture" width="700" />

<span style="color:white"><span style="color:#ff2777">**SRAM**</span> runs at 19 TB/s but has only 20 MB. <span style="color:#ff2777">**HBM**</span> runs at 1.5 TB/s with 40 GB—it's 10× slower but much larger. Standard attention writes the N×N matrix to HBM repeatedly. FlashAttention keeps the computation in SRAM.</span>



### >TILING ALGO

<span style="color:white">Now if we look at center of the figure, we can see how the tiling works. FlashAttention breaks the large <span style="color:#ff2777">**Q**</span>, <span style="color:#ff2777">**K**</span>, <span style="color:#ff2777">**V**</span> matrices into smaller <span style="color:#ff2777">**blocks (tiles)**</span> that do fit into fast SRAM. It uses two nested loops: the outer loop iterates over blocks of <span style="color:#ff2777">**K**</span> and <span style="color:#ff2777">**V**</span>, the inner loop iterates over blocks of <span style="color:#ff2777">**Q**</span>. For each block combination, compute attention entirely <span style="color:#ff2777">**on-chip**</span> and update the output.</span>

<span style="color:white">All operations (matrix multiply, masking, softmax, dropout) happen in SRAM. The N×N attention matrix never gets written to HBM.</span>

### BUT HOW DOES THIS WORK WITH SOFTMAX?

<span style="color:white">Softmax is applied per query and depends on all keys at once. To normalize correctly, you need the maximum score and the sum of exponentials over the full attention row. That's why FlashAttention uses an <span style="color:#ff2777">**online softmax**</span>: instead of computing the softmax in one pass over all keys, the algorithm processes key blocks <span style="color:#ff2777">**incrementally**</span> while maintaining two running values for each query: the current maximum score and the current sum of exponentials. When a new block of keys is processed, these statistics are updated, and previously accumulated values are <span style="color:#ff2777">**rescaled**</span> if needed.

<span style="color:white">In my code, for example, you can see this through the per-query variables <span style="color:#ff2777">`m`</span> and <span style="color:#ff2777">`l`</span>, which track the <span style="color:#ff2777">**running max**</span> and <span style="color:#ff2777">**sum of exponentials**</span> across key blocks.</span>


<span style="color:white">This online softmax produces the exact same result as a standard one, just the order in which the computation is performed changes.</span>

### >KERNEL FUSION & RECOMPUTATION

<span style="color:white">Because all intermediate values stay on-chip, FlashAttention fuses the entire attention computation into a <span style="color:#ff2777">** single kernel**</span>. This avoids repeated kernel launches and unnecessary synchronization between stages.

<span style="color:white">During the <span style="color:#ff2777">**backward pass**</span>, it does not store the attention matrix either. Instead, attention scores are <span style="color:#ff2777">**recomputed**</span> on the way using the same tiling strategy as in the forward pass. This trades extra computation for much lower memory traffic, which is a kinda nice trade-off for modern GPUs.</span>

<br>

<br>

<br>

## FLASHATTENTION 2

<span style="color:white">When you first hear about FlashAttention-2, you expect a big algorithmic change. There isn't one.

<span style="color:white">FlashAttention-2 computes the same thing as FlashAttention-1, with the same tiling and the same online softmax. The difference is not in the math but in how the computation is <span style="color:#ff2777">**scheduled**</span> on the GPU.</span>



### >LOOP REORDERING

<span style="color:white">The key change in FlashAttention-2 is a reordering of the loops.

<span style="color:white">FlashAttention-1 uses outer loop over <span style="color:#ff2777">**K/V**</span> blocks, inner loop over <span style="color:#ff2777">**Q**</span> blocks. This means each <span style="color:#ff2777">**Q**</span> block's output gets updated multiple times, once per <span style="color:#ff2777">**K/V**</span> block causing <span style="color:#ff2777">**scattered writes**</span> to HBM.

<span style="color:white">To fix this, FA2 uses outer loop over <span style="color:#ff2777">**Q**</span> blocks and inner loop over <span style="color:#ff2777">**K/V**</span> blocks. Each <span style="color:#ff2777">**Q**</span> block is loaded once, processed against all <span style="color:#ff2777">**K/V**</span> blocks while staying in SRAM, and written once when done.

<span style="color:white">So we have <span style="color:#ff2777">**sequential writes**</span> instead of scattered writes. GPUs <span style="color:#ff2777">**coalesce**</span> sequential memory accesses into larger and more efficient transactions. It's the difference between writing to consecutive addresses VS jumping around randomly.</span>

### >PARALLELIZATION

<span style="color:white">Reordering the loops also lets you <span style="color:#ff2777">**parallelize better**</span>.

<img src="../assets_lab/a1fig2.png" alt="flashattention 2 improvements" width="700" />

<span style="color:white">Look at this other beautiful figure from the paper showing work partitioning between warps. In FA1 (left), <span style="color:#ff2777">**K<sup>T</sup>**</span> is split across different <span style="color:#ff2777">**warps**</span> horizontally while <span style="color:#ff2777">**Q**</span> & <span style="color:#ff2777">**V**</span> are accessed by all warps. 
Parallelism happens mostly across heads and batches. One attention head per thread block means low GPU utilization, especially for long sequences with small batches.. This can lead to low <span style="color:#ff2777">**occupancy**</span> specially for long sequences and small batch sizes.

<span style="color:white">FA2 (right) changes this completely. Now <span style="color:#ff2777">**Q**</span> is split across warps vertically while <span style="color:#ff2777">**K<sup>T</sup>**</span> and <span style="color:#ff2777">**V**</span> are accessed by all warps. Multiple thread blocks can now cooperate on the same attention head, each handling different <span style="color:#ff2777">**Q**</span> blocks. More occupancy and keeps more <span style="color:#ff2777">**SMs**</span> busy, allowing the kernel to better saturate the GPU!!</span>

<br>

<span style="color:white">No algorithmic change but a good re-factoring around GPU execution.

<span style="color:white">FlashAttention-2 computes exactly the same attention as FlashAttention-1, the speedup comes entirely from better <span style="color:#ff2777">**GPU scheduling**</span>.</span>



<br>
<br>

<br>

Playing with attention kernels made me realize how much performance comes from these low-level choices. That’s why this whole attention kernel topic interests me, i want to explore megakernel attention next hehe.

<br>
<br>
---

<span style="color:white">LINK:
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [code here](https://github.com/jalexine/gpucodes)

<br>
