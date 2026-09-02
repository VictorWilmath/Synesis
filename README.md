# bodytracker

Single-webcam full-body tracking for VRChat, anchored on the headset.

## Why this is different

Most webcam FBT tools estimate a complete 3D skeleton from the camera alone,
then fight to align that arbitrary camera space with VRChat's play space. Depth
is their well-known weak point, and it follows directly from that choice: a
single camera cannot recover metric depth.

If you are already in VR, you do not have to. SteamVR hands you three metric,
drift-free 6DoF poses at 90 Hz — head and both hands — and that changes the
problem:

- **Scale and depth stop being ambiguous.** Solve the camera extrinsics once
  against the HMD and the head's metric 3D position is known every frame. Bone
  lengths constrain the rest.
- **The output is natively in VRChat's coordinate space.** No head-anchor
  realignment hack; `/tracking/trackers/head/*` becomes optional.
- **Only the hard-to-see joints need estimating:** hips, chest, knees, feet.

Making it robust on a cheap 720p webcam is then a *data* problem, not a
resolution problem — see [Training](#training).

## Requirements

- **Python 3.11 or 3.12.** Not 3.13+; the ONNX Runtime and PyTorch wheels lag
  new releases. The package technically installs on anything 3.11 or newer, but
  the `runtime` and `training` extras will not resolve on the newest versions.
- A VR headset with SteamVR running (this is VR-only; VRChat's OSC tracker
  endpoints feed its calibrated full-body IK system, which does not exist in
  desktop mode).
- A webcam that can see your whole body.
- NVIDIA GPU recommended but not required — the 2D model runs at 90+ FPS on a
  modern CPU, which keeps the GPU free for rendering.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[runtime,dev]"
pip install -e ".[gpu]"          # optional, for CUDA inference
```

On the training box:

```bash
pip install -e ".[training,dev]"
```

## Getting started

The phases are ordered so that the most opaque thing to debug — whether VRChat
will accept your trackers at all — is proven first, with no computer vision in
the way.

### 1. Prove the OSC wire

```bash
bodytracker-osc-smoke --pattern bob
```

This streams a canned, anatomically plausible 8-tracker pose to
`127.0.0.1:9000`. In VRChat: enable OSC (Options → OSC → Enable), then run
full-body calibration. If the avatar's hips and feet follow the synthetic
motion, the wire works. See `--help` for other patterns.

### 2. Calibrate

```bash
bodytracker calibrate-intrinsics     # one-time, needs a printed checkerboard
bodytracker calibrate-extrinsics     # wave the controllers around
bodytracker calibrate-body           # T-pose, estimates your bone lengths
```

Extrinsics calibration needs no checkerboard: the camera watches your head and
wrists while SteamVR reports where they actually are, and PnP recovers the
camera pose from those correspondences.

### 3. Run

```bash
bodytracker run
```

## How the depth ambiguity is actually resolved

Lifting 2D keypoints to 3D by bone lengths means solving, for each bone, where
a joint sits along its viewing ray given its parent and a known bone length.
That equation is a quadratic, and it has two roots: the joint can be nearer or
further than its parent and project to exactly the same pixel. Choosing between
them is the entire problem, and choosing wrong does not produce a small error,
it produces a limb pointing backwards.

Four priors do the choosing, in descending order of how much they can be
trusted:

1. **The headset**, which gives the head's true metric position, so the chain
   starts from a known point rather than a guessed one. It also supplies a
   rough facing direction, which is the only thing that can distinguish turned
   slightly left from turned slightly right.
2. **The previous frame**, which is a measurement of the actual pose rather
   than an assumption about it, and so outranks the two below whenever it
   exists.
3. **Gravity**, for bones that hang along it — the spine, the limbs. Only
   applied when one candidate is close to vertical outright; a subject bent
   forty degrees forward has two leaning candidates and straightening them up
   would be confidently wrong.
4. **Anatomy**, for the rest: shoulders sit on opposite sides of the neck, feet
   point the way the body faces, the little toe is on the outside.

On synthetic data with a correct body scale this recovers every joint to within
a few millimetres. What is left is a greedy, bone-at-a-time solver that commits
to each choice as it walks the skeleton, which is the main thing the trained
lifter should improve on.

## Configuration

`configs/default.toml` is the version-controlled baseline. Copy the keys you
want to change into `configs/local.toml`, which is gitignored and overlaid on
top.

The tracker role set is worth tuning. VRChat's own documentation notes that its
IK compensates better with fewer trackers, so the default is hip plus both
feet rather than all eight:

```toml
[osc]
roles = ["hip", "left_foot", "right_foot"]
```

## Training

The 2D front-end is off-the-shelf RTMPose. The custom piece is the **temporal
lifter**: it takes a window of 2D keypoints plus the known HMD and controller
poses and predicts metric 3D positions and orientations for the joints the
camera has to guess at.

Training data is [BEDLAM2.0](https://bedlam2.is.tuebingen.mpg.de/) — synthetic,
SMPL-X ground truth, and rendered natively at 1280x720, which happens to be
exactly the target domain. It requires registration and is non-commercial use
only. Do not download all 11 TB; `training/bedlam/subset.py` selects the
room-scale, static-camera, single-subject sequences that match a webcam setup.

Robustness on a cheap camera comes from **degradation augmentation** rather
than a bigger model: MJPEG compression artifacts, motion blur, Poisson-Gaussian
sensor noise, exposure hunting, white balance drift, rolling shutter skew and
FOV cropping. `training/augment/profile.py` measures your actual camera's noise
characteristics so the augmentation can be matched to it.

```bash
python -m training.bedlam.prep --config configs/train_lift.yaml
python -m training.train_lift --config configs/train_lift.yaml
python -m training.export_onnx --checkpoint checkpoints/lifter/best.pt
```

## Measuring whether it is actually good

You cannot beat existing tools without a number, and you have no mocap suit.
The trick is to **hold out a tracked device**: hide the headset from the lifter,
let it predict where your head is from the camera alone, then compare that
against what SteamVR actually reported for the same frame. SteamVR is accurate
to about a millimetre, so the difference is true metric error — on your
hardware, in your room, under your lighting, for free. The controllers give the
same test for the wrists.

```bash
bodytracker run --record recordings/session01   # tracks and records at the same time
bodytracker eval recordings/session01 --held-out
bodytracker compare recordings/baseline recordings/session01 --held-out
```

A session stores the pipeline's **inputs** — 2D keypoints and SteamVR poses —
not just its outputs. That is what makes it possible to re-run old sessions
through a new lifter months later and compare on identical frames, with no
camera and no headset attached:

```bash
bodytracker compare-lifters recordings/session01 --baseline geometric --candidate anchored
```

On a synthetic session that prints, and this is the Phase 2 result in one
table:

```
metric                          geometric         anchored     change
held-out head                     24.8 cm          24.9 cm      +0.4%
held-out left_hand                13.1 cm           6.0 cm     -54.3%  better
held-out right_hand               12.6 cm           6.4 cm     -49.1%  better
```

The wrists halve because the headset makes the skeleton metric. The head does
not move because withholding it is exactly what removes that advantage, which
is the point: **25 cm is what the camera alone is worth, and 6 cm is what the
camera plus a headset is worth.** Closing that gap without the headset is the
job of the trained lifter.

Averaged error is not the whole story, so `bodytracker eval` also reports:

- **Jitter** — movement while the headset says you are standing still.
- **Foot skate** — a foot on the ground sliding horizontally while you are not
  going anywhere. Gated on the headset, because a foot moving during a step is
  a person walking, not a tracking error.
- **Floor penetration** — how often and how far feet end up underground.
- **Latency** — capture to targets ready, broken down by stage.

Known gap: if the headset drops out mid-session the lifter falls all the way
back to guessing scale, even when both controllers are still reporting. It
should anchor to a wrist instead.

## Layout

```
src/bodytracker/
  geom/       coordinate spaces, rotations, Euler ZXY
  capture/    webcam I/O, exposure locking, intrinsics
  pose2d/     rtmlib wrapper, bbox tracking between detector runs
  vr/         SteamVR device poses via pyopenvr
  calib/      PnP extrinsics, body proportions
  lift/       2D -> metric 3D, HMD-conditioned
  solve/      bone-length and floor constraints, rotation derivation
  filters/    One Euro, rotation smoothing, hold-last-valid
  osc/        VRChat sender
  eval/       session recording, replay, held-out metrics
  app/        runtime loop, CLI, debug overlay
training/     BEDLAM2 prep, degradation augmentation, lifter, ONNX export
```

## A note on coordinate spaces

This is where projects like this die. Three conventions are in play:

| Space | Handedness | Up | Forward |
| --- | --- | --- | --- |
| OpenCV camera | right | -y | +z |
| SteamVR play | right | +y | -z |
| Unity / VRChat | left | +y | +z |

All conversions live in `src/bodytracker/geom/` and are unit-tested. Nothing
else in the codebase should be flipping signs by hand. Tracker rotations go out
as **Euler ZXY in degrees**, which is Unity's convention and what VRChat
expects.
