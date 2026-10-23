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
        "slide": "video_frames/slide_01.png",
        "text": "Did llama dot c-p-p really just make 125B MoE models four times faster? Kai recently dropped a deep dive analyzing the new MoE expert cache in PR 29887. But what actually happens when we test this on real hardware: an RTX 5060 Ti 16GB, running the official 24-gigabyte Nemotron 3.5 Bolt?"
    },
    {
        "id": "scene_02",
        "slide": "video_frames/slide_02.png",
        "text": "Here is the architectural reality. Nemotron 3.5 Bolt packs 496 routed experts per layer across 40 layers. That's 19,840 total experts. On a 16-gigabyte card, a 6-gigabyte cache only fits about 4,500 slices. That means you're looking at a brutal 78 percent cache miss rate."
    },
    {
        "id": "scene_03",
        "slide": "video_frames/slide_03.png",
        "text": "And this is where Kai's warning at timestamp 23:17 becomes critical. In llama dot c-p-p, every single miss triggers a synchronous cudaStreamWaitEvent. The Tensor Cores halt, execution freezes, and decode speed completely collapses down to just 3.5 tokens per second."
    },
    {
        "id": "scene_04",
        "slide": "video_frames/slide_04.png",
        "text": "MinnieTheMoEcher takes a fundamentally different approach: asynchronous double-buffered DMA. While Layer N computes Mamba-2 SSM and dense Attention on the Tensor Cores, Layer N plus 1's experts are pre-streamed in parallel from pinned system memory. The PCIe transfer latency is completely masked behind computation."
    },
    {
        "id": "scene_05",
        "slide": "video_frames/slide_05.png",
        "text": "The result? Minnie hits 13.51 tokens per second decode—almost 4 times faster than llama dot c-p-p's MoE cache—with zero percent CPU burn. 70-billion parameter MoE inference running smoothly on a sub-500-dollar consumer GPU is now a reality."
    }
]

FFMPEG = r"C:\Program Files (x86)\CoolerMaster\MasterPlus\ffmpeg.exe"
BASE_DIR = r"E:\dev\MinnieTheMoEcher"
AUDIO_DIR = os.path.join(BASE_DIR, "neural_audio")
SCENES_DIR = os.path.join(BASE_DIR, "neural_scenes")

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
        # Add 0.5s pause after speech
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
        
    final_mp4 = os.path.join(BASE_DIR, "kai_style_moe_benchmark.mp4")
    print(f"Concatenating into final MP4: {final_mp4}")
    concat_cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        final_mp4
    ]
    subprocess.run(concat_cmd, check=True)
    print("Done! Neural video successfully generated!")

if __name__ == "__main__":
    asyncio.run(synthesize_all())
    render_scenes()
