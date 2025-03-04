import patato as pat
import numpy as np
import glob
from pathlib import Path

in_folder = Path("Raw Data")
out_folder = Path("Data")

for f in glob.glob(str(in_folder / "SKIN*")):
    skin_id = Path(f).stem
    print("Working on", skin_id)
    out_path = Path(out_folder / skin_id)
    out_path.mkdir(parents=True, exist_ok=True)

    for scan in glob.glob(f + "/**/*.msot", recursive=True):
        scan_id = Path(scan).stem
        output_file = out_path / (scan_id + ".hdf5")
        if Path(output_file).exists():
            continue
        print("    Working on", scan_id)
        pa_ithera = pat.iTheraMSOT(Path(scan).parent)
        padata_ithera = pat.PAData(pa_ithera)
        if padata_ithera.get_ultrasound():
            us_ithera = pat.PAData(pa_ithera).get_ultrasound()[:, 0].raw_data
            loss = np.sum((us_ithera[1:] - us_ithera[:-1]) ** 2, axis=(1, 2, 3))
            frame = np.argmin(loss[:-2])  # exclude last 2 frames?
        else:
            print("    US not acquired: taking central frame", scan_id)
            frame = padata_ithera.shape[0] // 2
        pa_output = pat.PAData(pa_ithera)[frame : frame + 1]
        print(f"    Will write to {output_file}, with shape{pa_output.shape}")
        pa_output.scan_reader.save_to_hdf5(output_file)
