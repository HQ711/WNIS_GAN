"""Example: reading `spherical_grid.npz` using display-aligned coordinates.

Preferred coordinates in this reader:
- `display_up_down_deg`
- `display_left_right_deg`

Legacy world-coordinate keys may still exist in the NPZ for backward compatibility,
but this example uses display coordinates as the primary interpretation space.

Human body region heuristic (display coordinates):
- Head: range 0.90m ~ 1.00m, and 0 deg <= `display_up_down_deg` <= 20 deg
- Chest: range 0.85m ~ 0.95m, and `display_up_down_deg` < 0 deg
- Arm: range 0.40m ~ 0.70m, and `display_up_down_deg` < 0 deg

Note: this is an engineering heuristic for interpretation, not strict anatomy GT.
"""

from pathlib import Path
import numpy as np


def load_spherical_grid(npz_path):
    data = np.load(npz_path)

    has_display = 'display_up_down_deg' in data.files and 'display_left_right_deg' in data.files
    if has_display:
        display_ud = data['display_up_down_deg']
        display_lr = data['display_left_right_deg']
    else:
        print('Warning: display-aligned keys not found, falling back to legacy world-coordinate keys.')
        display_ud = data['world_up_down_deg']
        display_lr = data['world_left_right_deg']

    return {
        'velocity': data['velocity'],
        'display_up_down_deg': display_ud,
        'display_left_right_deg': display_lr,
        'world_up_down_deg': data['world_up_down_deg'] if 'world_up_down_deg' in data.files else None,
        'world_left_right_deg': data['world_left_right_deg'] if 'world_left_right_deg' in data.files else None,
        'board_elevation_deg': data['board_elevation_deg'] if 'board_elevation_deg' in data.files else None,
        'board_azimuth_deg': data['board_azimuth_deg'] if 'board_azimuth_deg' in data.files else None,
        'power_db': data['power_db'] if 'power_db' in data.files else None,
        'frame_indices': data['frame_indices'],
        'range_vals': data['range_vals'],
    }


def build_region_masks(range_vals, display_ud_frame):
    range_grid = np.broadcast_to(range_vals[:, None], display_ud_frame.shape)

    head_mask = (
        (range_grid >= 0.90)
        & (range_grid <= 1.00)
        & (display_ud_frame >= 0.0)
        & (display_ud_frame <= 20.0)
    )
    chest_mask = (
        (range_grid >= 0.85)
        & (range_grid <= 0.95)
        & (display_ud_frame < 0.0)
    )
    arm_mask = (
        (range_grid >= 0.40)
        & (range_grid <= 0.70)
        & (display_ud_frame < 0.0)
    )

    return {
        'head': head_mask,
        'chest': chest_mask,
        'arm': arm_mask,
    }


def main():
    npz_path = Path('/Users/yiwei/Dropbox/driver_globecom/data/3.23.2026/yi/exp5/spherical_grid.npz')
    if not npz_path.exists():
        raise FileNotFoundError(f'NPZ not found: {npz_path}')

    d = load_spherical_grid(str(npz_path))

    vel = d['velocity']
    display_ud = d['display_up_down_deg']
    display_lr = d['display_left_right_deg']
    frame_ids = d['frame_indices']
    range_vals = d['range_vals']

    F, R, A = vel.shape
    print(f'Frames: {F}')
    print(f'Range bins: {R} ({range_vals[0]:.2f}m ~ {range_vals[-1]:.2f}m)')
    print(f'Angle bins: {A}')
    print('Axis naming used by this reader: display_up_down_deg / display_left_right_deg')
    print('Body-region heuristic uses display coordinates only.')
    print('  Head : 0.90~1.00 m and 0~20 deg')
    print('  Chest: 0.85~0.95 m and display_up_down_deg < 0 deg')
    print('  Arm  : 0.40~0.70 m and display_up_down_deg < 0 deg')

    frame_idx = 0
    range_idx = min(5, R - 1)
    angle_idx = min(10, A - 1)

    ud = display_ud[frame_idx, range_idx, angle_idx]
    lr = display_lr[frame_idx, range_idx, angle_idx]
    vv = vel[frame_idx, range_idx, angle_idx]

    print('\n--- Single bin (display coordinates) ---')
    print(f'Frame {frame_ids[frame_idx]}, range={range_vals[range_idx]:.3f}m, angle_bin={angle_idx}')
    print(f'display_up_down_deg={ud:.3f}, display_left_right_deg={lr:.3f}, velocity={vv:.3f} m/s')

    min_speed = 0.3
    moving_mask = np.abs(vel[frame_idx]) > min_speed
    moving_count = int(np.sum(moving_mask))
    total_count = vel[frame_idx].size
    print(f'\nMoving bins in frame {frame_ids[frame_idx]} (|v|>{min_speed} m/s): {moving_count}/{total_count}')

    frame_region_masks = build_region_masks(range_vals, display_ud[frame_idx])
    print('\n--- Region counts in display coordinates ---')
    for region_name, region_mask in frame_region_masks.items():
        region_total = int(np.sum(region_mask))
        region_moving = int(np.sum(region_mask & moving_mask))
        print(f'{region_name}: moving={region_moving}, total={region_total}')

    table = np.column_stack([
        np.broadcast_to(range_vals[:, None], (R, A)).reshape(-1),
        display_ud[frame_idx].reshape(-1),
        display_lr[frame_idx].reshape(-1),
        vel[frame_idx].reshape(-1),
        frame_region_masks['head'].reshape(-1).astype(np.int32),
        frame_region_masks['chest'].reshape(-1).astype(np.int32),
        frame_region_masks['arm'].reshape(-1).astype(np.int32),
    ])

    output_csv = npz_path.with_name('frame0_display_table.csv')
    np.savetxt(
        output_csv,
        table,
        delimiter=',',
        header='range_m,display_up_down_deg,display_left_right_deg,velocity,is_head,is_chest,is_arm',
        comments='',
        fmt='%.4f',
    )
    print(f'\nExported: {output_csv}')


if __name__ == '__main__':
    main()
