import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])
df_manual = pd.read_csv("femur_manual_cut_plane/metrics_log/metric-log.csv")
df_predicted = pd.read_csv("2d-3d-benchmark/ceio7qj7/evaluation/morphometry-log.csv")
df_generalized_metrics = pd.read_csv(
    "2d-3d-benchmark/ceio7qj7/evaluation/metric-log.csv"
)
print(df_manual)
print(df_predicted)

fig, ax = plt.subplots(1, 3)
for subject_id, fhr, fna, fho in df_predicted.to_numpy():
    subject_id, fhr_gt, fna_gt, fho_gt = (
        df_manual[df_manual["subject-id"] == subject_id].to_numpy().flatten()
    )
    if fna_gt < 90.0:
        fna_gt = 180.0 - fna_gt
    if fna < 90.0:
        fna = 180.0 - fna
    fhr_diff = fhr - fhr_gt
    fna_diff = fna - fna_gt
    fho_diff = fho - fho_gt
    dsc = (
        df_generalized_metrics[
            df_generalized_metrics["subject-id"].str.contains(subject_id.split("_")[0])
        ]
        .iloc[:1]["DSC"]
        .values[:1][0]
    )
    asd = (
        df_generalized_metrics[
            df_generalized_metrics["subject-id"].str.contains(subject_id.split("_")[0])
        ]
        .iloc[:1]["ASD"]
        .values[:1][0]
    )
    hd95 = (
        df_generalized_metrics[
            df_generalized_metrics["subject-id"].str.contains(subject_id.split("_")[0])
        ]
        .iloc[:1]["HD95"]
        .values[:1][0]
    )
    print(
        f"{subject_id:30s} {fhr_diff:7.2f},{fna_diff:7.2f},{fho_diff:7.2f} {dsc:7.2f} {asd:7.2f} {hd95:7.2f}"
    )

    ax[0].scatter(dsc, fhr_diff)
    ax[1].scatter(dsc, fna_diff)
    ax[2].scatter(dsc, fho_diff)
ax[0].set_xlabel("DSC")
ax[0].set_ylabel("FHR(mm)")
ax[1].set_xlabel("DSC")
ax[1].set_ylabel("FNA(degree)")
ax[2].set_xlabel("DSC")
ax[2].set_ylabel("FHO(mm)")
ax[0].set_xticks([0.95])
ax[1].set_xticks([0.95])
ax[2].set_xticks([0.95])
plt.tight_layout()
plt.savefig("visualizations/ceio7qj7_dice_vs_femur.png")
plt.show()
