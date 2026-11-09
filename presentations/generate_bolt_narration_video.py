import os
import subprocess
import asyncio
import edge_tts

VOICE = "en-US-ChristopherNeural"
RATE = "+3%"
PITCH = "+0Hz"

SCENES = [
    {
        "id": "scene_01",
        "slide": "bolt_video_frames/slide_01.png",
        "text": "Thank you, NVIDIA! For years, running a true 70-billion parameter model on a sub-500-dollar consumer GPU felt like science fiction. But NVIDIA's release of the 24-gigabyte Nemotron 3.5 Bolt changes everything. By combining Mamba-2 state space layers with a massive 496-expert LatentMoE, high-end AI inference is finally within reach of everyday developers."
    },
    {
        "id": "scene_02",
        "slide": "bolt_video_frames/slide_02.png",
        "text": "Nemotron 3.5 Bolt is not a toy. It has 74 billion parameters, routing 9 active experts per token across 40 layers. That equals 19,840 total routed experts. A 16-gigabyte RTX 5060 Ti cannot fit this entirely in VRAM. It demands a flawless multi-tier memory architecture between GPU VRAM and system DDR5 RAM."
    },
    {
        "id": "scene_03",
        "slide": "bolt_video_frames/slide_03.png",
        "text": "We benchmarked this head-to-head against the latest llama dot c-p-p release on an RTX 5060 Ti 16GB. Standard full offload stalled below 1 token per second due to Windows driver paging thrash. Llama dot c-p-p's newly released MoE SLRU cache reached 3.5 tokens per second. Meanwhile, MinnieTheMoEcher delivered 13.51 tokens per second decode and 21.5 tokens prefill: a massive 3.86x speedup."
    },
    {
        "id": "scene_04",
        "slide": "bolt_video_frames/slide_04.png",
        "text": "Why is Minnie so much faster? In llama dot c-p-p, every cache miss halts the GPU, causing 48 sequential waits per token. Minnie eliminates this with asynchronous double-buffered DMA. While Layer N computes Mamba-2 SSM and Attention on Tensor Cores, Layer N plus 1's experts are pre-streamed in parallel from pinned host RAM. PCIe latency is completely masked behind computation."
    },
    {
        "id": "scene_05",
        "slide": "bolt_video_frames/slide_05.png",
        "text": "So what's next? We are already bringing Nemotron 3.5 Bolt to Apple Silicon using native Apple Metal and hardware MPP tiled dequantization. With unified memory architecture and up to 128 gigabytes of zero-copy RAM, Mac users will soon experience true 70B MoE intelligence with zero PCIe bus bottlenecks. Stay tuned!"
    }
]

FFMPEG = r"C:\Program Files (x86)\CoolerMaster\MasterPlus\ffmpeg.exe"
BASE_DIR = r"E:\dev\MinnieTheMoEcher"
AUDIO_DIR = os.path.join(BASE_DIR, "bolt_neural_audio")
SCENES_DIR = os.path.join(BASE_DIR, "bolt_neural_scenes")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)

async def synthesize_all():
    print(f"Synthesizing neural speech using voice: {VOICE}...")
    for scene in SCENES:
        mp3_path = os.path.join(AUDIO_DIR, f"{scene['id']}.mp3")
        print(f"Generating audio for {scene['id']}...")
        communicate = edge_tts.Communicate(scene["text"], VOICE, rate=RATE, pitch=PITCH)
        await communicate.save(mp3_path)
        print(f"Saved: {mp3_path}")

def render_scenes():
    concat_list = os.path.join(SCENES_DIR, "concat.txt")
    lines = []
    
    for i, scene in enumerate(SCENES, start=1):
        slide_img = os.path.join(BASE_DIR, scene["slide"])
        audio_mp3 = os.path.join(AUDIO_DIR, f"{scene['id']}.mp3")
        scene_mp4 = os.path.join(SCENES_DIR, f"scene_{i:02d}.mp4")
        
        print(f"Rendering Video Scene {i}...")
        cmd = [
            FFMPEG, "-y",
            "-loop", "1", "-i", slide_img,
            "-i", audio_mp3,
            "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            scene_mp4
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        norm_path = scene_mp4.replace("\\", "/")
        lines.append(f"file '{norm_path}'")
        
    with open(concat_list, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
        
    final_mp4 = os.path.join(BASE_DIR, "nemotron_bolt_video.mp4")
    print(f"Concatenating into final MP4: {final_mp4}")
    concat_cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        final_mp4
    ]
    subprocess.run(concat_cmd, check=True)
    print("Done! Nemotron Bolt video successfully generated!")

if __name__ == "__main__":
    asyncio.run(synthesize_all())
    render_scenes()
