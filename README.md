# Interview Task: GAN for Single-Target Motion Relation Learning

## Background

This interview package contains radar-based motion recordings from a single
person in a seated posture: `P1`.

Your task is to build a GAN-based solution that learns the relationship between
different body regions from P1's data only.

More specifically, the target problem is:

- Input: motion features from `head + arms`
- Output: motion features from `torso`

You do **not** need to do identity classification across different people.
This is a **single-person modeling task**. The purpose is to evaluate whether
you can build a clean pipeline that learns the mapping from head/arms motion to
torso motion and verify that the learned relation works on held-out P1 data.

## Folder Structure

```text
interview_gan_p1/
├── README.md
├── data/
│   ├── train/
│   │   ├── p1_1_spherical_grid.npz
│   │   ├── p1_2_spherical_grid.npz
│   │   └── p1_3_spherical_grid.npz
│   └── val/
│       └── p1_4_spherical_grid.npz
└── sample/
    ├── sample_reader.py
```

## Data

Each `.npz` file contains frame-wise radar-related arrays such as:

- `velocity`
- `display_up_down_deg`
- `display_left_right_deg`
- `world_up_down_deg`
- `world_left_right_deg`
- `board_elevation_deg`
- `board_azimuth_deg`
- `power_db`
- `frame_indices`
- `range_vals`

Please see `sample/sample_reader.py` for a
minimal format example.

## Required Task

Please implement a pipeline that includes:

1. data loading,
2. feature construction,
3. GAN model definition,
4. training,
5. validation on held-out P1 data.

Your model should use `head + arms` features to predict or generate `torso`
features.

## What We Expect

We care about:

- whether your data pipeline is logically sound,
- whether your model design is reasonable,
- whether the code is runnable and organized,
- whether your evaluation is meaningful,
- whether you can explain the design choices clearly.

We do **not** expect perfect performance. We are more interested in whether you
can reason clearly about the problem and build a working end-to-end solution.

## Suggested Deliverables

Please provide:

1. runnable code,
2. a short explanation of your design,
3. training and validation results,
4. any plots or analysis that help demonstrate whether the model learned the
   intended relationship.

## Notes

- Please use only the P1 data included in this folder.
- You are free to simplify the feature design as long as the logic is clear.
- You may use any reasonable GAN variant, but please explain your choice.
