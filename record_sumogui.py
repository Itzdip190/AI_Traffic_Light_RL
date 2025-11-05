# record_sumogui.py
import os
import time
import imageio
import traci
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- SUMO CONFIGURATION ---
SUMO_CMD = ["sumo-gui", "-c", "sumo_project/cross.sumocfg"]  # update if needed
OUT_DIR = "video_frames"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

STEP_PER_FRAME = 2
MAX_STEPS = 1000
VIDEO_PATH = "output/sumo_simulation.mp4"


# --- SAFE OVERLAY FUNCTION ---
def overlay_text(img_path, text):
    """Overlay text safely after confirming the image exists."""
    timeout = 1.0  # seconds to wait for file to appear
    start = time.time()

    # Wait until the file is written
    while not os.path.exists(img_path):
        if time.time() - start > timeout:
            print(f"⚠️ Skipping frame — file not found: {img_path}")
            return
        time.sleep(0.05)

    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"⚠️ Failed to open image {img_path}: {e}")
        return

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Compute box size dynamically
    text_w, text_h = draw.textsize(text, font=font)
    padding = 8
    box = [5, 5, 5 + text_w + padding * 2, 5 + text_h + padding * 2]
    draw.rectangle(box, fill=(0, 0, 0, 160))
    draw.text((5 + padding, 5 + padding), text, fill=(255, 255, 255, 255), font=font)
    img.save(img_path)


# --- MAIN SIMULATION FUNCTION ---
def main():
    print("🚦 Launching SUMO GUI...")
    try:
        traci.start(SUMO_CMD)
    except FileNotFoundError:
        print("❌ ERROR: 'sumo-gui' not found. Check if SUMO is installed and added to PATH.")
        print("Try running 'sumo-gui' manually or provide full path in SUMO_CMD.")
        raise
    except Exception as e:
        print("❌ traci.start() failed:", e)
        raise

    # Allow GUI to settle
    time.sleep(1.0)

    # Get view ID for screenshots
    try:
        views = traci.gui.getIDList()
        view_id = views[0] if views else ""
    except Exception:
        view_id = ""

    print(f"✅ SUMO started successfully. View ID → '{view_id}'")
    frame_paths = []

    # --- Simulation Loop ---
    try:
        for step in tqdm(range(MAX_STEPS)):
            traci.simulationStep()

            if step % STEP_PER_FRAME == 0:
                img_path = os.path.join(OUT_DIR, f"frame_{step:06d}.png")

                # Capture screenshot safely
                try:
                    traci.gui.screenshot(view_id, img_path)
                    time.sleep(0.1)  # allow time for SUMO to save frame
                except Exception as e:
                    try:
                        traci.gui.screenshot("", img_path)
                        time.sleep(0.1)
                    except Exception as e2:
                        print(f"⚠️ Screenshot failed at step {step}: {e2}")
                        continue

                # Collect traffic light states
                try:
                    tls_ids = traci.trafficlight.getIDList()
                    state_txts = []
                    for t in tls_ids:
                        try:
                            s = traci.trafficlight.getRedYellowGreenState(t)
                            state_txts.append(f"{t}:{s}")
                        except Exception:
                            pass
                    state_txt = " ".join(state_txts) if state_txts else ""
                except Exception:
                    state_txt = ""

                # Overlay text info
                overlay_text(img_path, f"step {step} | time {traci.simulation.getTime():.1f} {state_txt}")
                frame_paths.append(img_path)

    finally:
        # Ensure SUMO closes properly
        try:
            traci.close()
        except Exception:
            pass

    # --- Assemble Video ---
    if not frame_paths:
        print("⚠️ No frames were captured. Video will not be created.")
        return

    print("🎞️ Assembling video...")
    writer = imageio.get_writer(VIDEO_PATH, fps=10)
    for f in frame_paths:
        try:
            writer.append_data(imageio.imread(f))
        except Exception as e:
            print(f"⚠️ Skipped frame {f}: {e}")
    writer.close()

    print(f"✅ Simulation video saved to → {VIDEO_PATH}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    main()
