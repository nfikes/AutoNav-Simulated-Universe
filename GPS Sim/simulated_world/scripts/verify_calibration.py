"""Calibration verification: sweep true_heading for each field datapoint,
find the value that places sim true_pos closest to the recorded field end,
and report agent termination conditions."""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import gps_sim_gui as g
import numpy as np

EARTH_R = 6378137.0
def latlon_to_m(lat, lon, lat0, lon0):
    de = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    dn = math.radians(lat - lat0) * EARTH_R
    return de, dn

DATAPOINTS = [
    dict(name="DP1",
         start=(37.23000, -80.42498),
         goal =(37.23027, -80.42504),
         end  =(37.22984, -80.42516)),
    dict(name="DP2",
         start=(37.23038, -80.42492),
         goal =(37.23027, -80.42504),
         end  =(37.23051, -80.42493)),
]

g._apply_real_overrides()

def run_sim(th_deg, gx, gy, max_steps=5000):
    rng = np.random.default_rng(42)
    cm = g.Costmap(obstacles=[], projectors=[])
    sim = g.GPSWaypointSim(cm, (0.0, 0.0), math.radians(th_deg), (gx, gy), rng,
                           roofs=[], projectors=[], jammers=[], foliage=[], spoofers=[])
    n = 0
    while n < max_steps:
        cont = sim.step()
        n += 1
        if not cont:
            break
    return sim, n

results = []
for dp in DATAPOINTS:
    s_lat, s_lon = dp['start']
    g_lat, g_lon = dp['goal']
    e_lat, e_lon = dp['end']
    gx, gy = latlon_to_m(g_lat, g_lon, s_lat, s_lon)
    ex, ey = latlon_to_m(e_lat, e_lon, s_lat, s_lon)
    print(f"\n=== {dp['name']} ===")
    print(f"  goal_world (m): ({gx:.2f}, {gy:.2f})  field_end (m): ({ex:.2f}, {ey:.2f})")
    print(f"  goal range from start: {math.hypot(gx, gy):.2f} m  end range: {math.hypot(ex, ey):.2f} m")

    # Coarse sweep
    best = None
    for th_deg in range(0, 360, 5):
        sim, steps = run_sim(th_deg, gx, gy)
        sex, sey = sim.true_pos
        err = math.hypot(sex - ex, sey - ey)
        if best is None or err < best['err']:
            best = dict(th_deg=th_deg, err=err, steps=steps, sim=sim, sex=sex, sey=sey)
    print(f"  coarse best: th={best['th_deg']} deg, err={best['err']:.2f} m, steps={best['steps']}, arrived={best['sim'].arrived}")

    # Fine sweep around coarse best (+/- 5 deg, 1 deg step)
    coarse_best = best['th_deg']
    for th_deg in range(coarse_best - 5, coarse_best + 6):
        th = th_deg % 360
        sim, steps = run_sim(th, gx, gy)
        sex, sey = sim.true_pos
        err = math.hypot(sex - ex, sey - ey)
        if err < best['err']:
            best = dict(th_deg=th, err=err, steps=steps, sim=sim, sex=sex, sey=sey)

    sim = best['sim']
    odom_err = math.hypot(sim.odom[0] - sim.goal_world[0],
                          sim.odom[1] - sim.goal_world[1])
    cg = sim.intermediate_goal_world()
    cand_dist = math.hypot(cg[0] - sim.true_pos[0], cg[1] - sim.true_pos[1])

    print(f"  FINE best: th={best['th_deg']} deg, err={best['err']:.3f} m")
    print(f"    arrived={sim.arrived}, steps={best['steps']} (cap=5000)")
    print(f"    sim true_pos = ({sim.true_pos[0]:.2f}, {sim.true_pos[1]:.2f})")
    print(f"    field_end    = ({ex:.2f}, {ey:.2f})")
    print(f"    sim.odom = ({sim.odom[0]:.3f}, {sim.odom[1]:.3f})")
    print(f"    sim.goal_world = ({sim.goal_world[0]:.3f}, {sim.goal_world[1]:.3f})")
    print(f"    odom_err_to_goal_world = {odom_err:.3f} m")
    print(f"    candidate_goal_world = ({cg[0]:.2f}, {cg[1]:.2f})")
    print(f"    candidate_to_agent_dist = {cand_dist:.3f} m")
    results.append(dict(name=dp['name'], best_heading=best['th_deg'], steps=best['steps'],
                        arrived=sim.arrived, err=best['err'],
                        odom_err=odom_err, cand_dist=cand_dist))

print("\n\n=== SUMMARY TABLE ===")
print(f"{'DP':<5}{'heading':>9}{'steps':>8}{'arrived':>9}{'err_field':>12}{'odom_err':>12}{'cand_dist':>12}")
for r in results:
    print(f"{r['name']:<5}{r['best_heading']:>9}{r['steps']:>8}{str(r['arrived']):>9}{r['err']:>12.3f}{r['odom_err']:>12.3f}{r['cand_dist']:>12.3f}")
