# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Lê Anh Duy
- MSSV: 2A202602723
- Email: duyla25clc@gmail.com
- Link repo (fork): https://github.com/Le-Anh-Duy/K4-L2L3-DAY23-LeAnhDuy-2A202602723-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): d8d52f9

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `precision = 0.9701 (0.9700934579439252)`, `recall = 0.7004 (0.7004048582995951)`, `tp = 519`, `fp = 16`, `fn = 222`
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse = 0.15032268781360134 m`, `matches = 502`, `sum_sq_err = 11.343649056695735`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.522613065326633`
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse = 0.1358667883353908 m`, `matches = 502`, `sum_sq_err = 9.26681165463209`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.522613065326633`
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở cả hai chế độ (`lidar` và `fused`), số lượng cặp ghép `matches` (502), số `ghost_track_frames` (0), số `missed_gt_frames` (239) và `mean_confirmed_tracks` (~2.52 track/frame) là hoàn toàn giống hệt nhau. Điều này minh chứng cho tính đúng đắn của thiết kế **track-then-fuse**: chỉ có LiDAR mới quyết định vòng đời track (tạo mới, cộng/trừ score, xác nhận và xóa track trong `track_management.py:manage_tracks`). Camera hoàn toàn không can thiệp vào lifecycle nên không sinh thêm ghost track (ghost = 0) và cũng không làm rớt mất track đã có.
  Về mặt độ chính xác vị trí 3D, chế độ `fused` đạt $RMSE = 0.1359\text{ m}$, tốt hơn so với chế độ `lidar` ($RMSE = 0.1503\text{ m}$), với mức chênh lệch $RMSE_{\text{fused}} - RMSE_{\text{lidar}} = -0.0145\text{ m} \le 0.05\text{ m}$ (đáp ứng tiêu chí chấm của RUBRIC). Tổng bình phương sai số `sum_sq_err` giảm từ $11.34\text{ m}^2$ xuống $9.27\text{ m}^2$ (giảm khoảng 18.3%), chứng tỏ việc kết hợp thông tin góc quan sát và phép chiếu 2D của camera thông qua Jacobian pinhole trong bước EKF update đã giúp tinh chỉnh và làm mịn ước lượng vị trí đối tượng.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - **Véc-tơ đo $z$:** LiDAR đo trực tiếp vị trí 3D trong không gian vật lý $z = [x, y, z]^T \in \mathbb{R}^3$ (tâm hộp 3D từ point cloud). Camera đo tọa độ điểm ảnh 2D trên mặt phẳng ảnh $z = [u, v]^T \in \mathbb{R}^2$ thu được từ phép chiếu phối cảnh pinhole.
   - **Ma trận hiệp phương sai nhiễu đo $R$:** LiDAR có $R \in \mathbb{R}^{3 \times 3} = \text{diag}(\sigma_{\text{lidar\_x}}^2, \sigma_{\text{lidar\_y}}^2, \sigma_{\text{lidar\_z}}^2)$ tính theo đơn vị mét vuông ($m^2$). Camera có $R \in \mathbb{R}^{2 \times 2} = \text{diag}(\sigma_{\text{cam\_i}}^2, \sigma_{\text{cam\_j}}^2)$ tính theo đơn vị pixel bình phương ($\text{px}^2$).
   - **Mô hình đo và Jacobian $H$:** LiDAR là quan sát tuyến tính trực tiếp ($H$ cố định $3 \times 6$), trong khi Camera là quan sát phi tuyến qua phép chiếu pinhole $h(x)$, do đó ma trận đo $H$ ($2 \times 6$) là ma trận Jacobian phụ thuộc trạng thái ước lượng $x$ thông qua đạo hàm chuỗi (chain rule: đạo hàm chiếu pinhole nhân ma trận xoay `veh_to_sens`).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Khoảng cách Mahalanobis ($d^2 = \gamma^T S^{-1} \gamma$) chuẩn hóa sai số đo (innovation $\gamma$) dựa trên độ bất định kết hợp $S = H P H^T + R$. Đại lượng $d^2$ tuân theo phân phối Chi-bình phương ($\chi^2$) với bậc tự do bằng số chiều đo $dim\_meas$.
   - Cần gating trước khi gán để:
     1. Loại bỏ các phép đo ngoại lai (outliers) hoặc các đối tượng quá xa phân bố xác suất của track, ngăn chặn việc cập nhật nhầm gây méo mó ma trận hiệp phương sai $P$ và trạng thái $x$ của bộ lọc Kalman.
     2. Giảm không gian tìm kiếm trong thuật toán ghép greedy (các cặp vượt ngưỡng $\chi^2$ bị gán chi phí $\infty$).
     3. Đảm bảo tính nhất quán thống kê, tránh hiện tượng track bị "nhảy" sang các phương tiện lân cận khi đi gần nhau trong không gian hẹp.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline là **track-then-fuse**: Hệ thống duy trì duy nhất một danh sách track ở mức đối tượng (track level). Trong mỗi frame, hệ thống thực hiện `predict` một lần duy nhất cho mọi track, sau đó lần lượt chạy EKF update theo từng sensor: cập nhật với LiDAR trước (`associate_and_update` cho LiDAR), rồi cập nhật bổ sung với Camera (`associate_and_update` cho Camera) trên cùng các track đó nếu đối tượng nằm trong tầm nhìn camera.
   - Minh chứng trong code và log của `fusion-run-lab` (`platform/fusion_lab/scripts/run_lab.py:run_frame`):
     Mỗi frame chỉ gọi `filter_obj.predict(track)` một lần cho toàn bộ track, sau đó gọi liên tiếp `associate_and_update(..., lidar)` rồi `associate_and_update(..., camera)`. Hoàn toàn không có bước tiền xử lý hợp nhất dữ liệu cảm biến thô (Point Cloud + Ảnh RGB) trước khi đưa vào mạng phát hiện (như fuse-then-track).

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Khi camera bị lệch extrinsic (vị trí/góc quay camera so với xe) hoặc intrinsic (tiêu cự/tâm quang học), hàm chiếu $h(x)$ sẽ dự đoán sai vị trí pixel của đối tượng trên ảnh.
   - **Triệu chứng trên Innovation ($\gamma = z - h(x)$):**
     1. Giá trị trung bình của innovation không còn bằng 0 ($\mathbb{E}[\gamma] \neq 0$) mà xuất hiện độ lệch hệ thống (systematic bias / non-zero residual offset) theo một hướng nhất định.
     2. Biên độ norm của innovation tăng vọt, dẫn đến khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ tăng mạnh.
   - **Hậu quả:** Nếu $d^2$ vượt ngưỡng cổng $\chi^2$, phép đo sẽ bị loại bỏ hoàn toàn (không thể update camera). Nếu $d^2$ vẫn nằm trong cổng, EKF update sẽ kéo sai lệch trạng thái $x$ của track, gây giật vị trí và làm tăng đáng kể sai số $RMSE$.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - **Cần sensor tường minh ở frame rỗng:** Khi một frame không có phép đo nào (`meas_list` rỗng), hệ thống vẫn phải kết thúc lượt bằng `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. Việc truyền sensor tường minh giúp `TrackManager` xác định đây là lượt của LiDAR hay Camera: nếu là lượt LiDAR, các track đang trong tầm quan sát (in FOV) nhưng không được gán đo sẽ bị tính là miss (bị trừ điểm tồn tại $1/\text{window}$ và xem xét xóa). Nếu là lượt Camera, hàm sẽ bỏ qua không thay đổi score.
   - **Vì sao LiDAR quyết định lifecycle còn Camera chỉ EKF update:** LiDAR cung cấp trực tiếp tọa độ không gian 3D chính xác, trường nhìn rộng 360 độ quanh xe và ít phụ thuộc vào điều kiện ánh sáng. Ngược lại, Camera trong bài lab chỉ có góc nhìn phía trước (FRONT FOV), đo đạc là phép chiếu 2D thiếu trực tiếp chiều sâu, dễ bị che khuất. Nếu camera can thiệp vào lifecycle, những chiếc xe đi ra ngoài tầm nhìn camera phía trước hoặc xe ở phía sau sẽ bị trừ điểm và xóa nhầm, hoặc tạo ra nhiều track rác (ghost). Do đó, chỉ LiDAR mới có thẩm quyền khởi tạo, tính điểm và xóa track; Camera chỉ hỗ trợ tinh chỉnh (refine) trạng thái hình học trong EKF update.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Điều kiện xác nhận (confirmed):** Track được khởi tạo với điểm ban đầu $\text{score} = 1/\text{window} = 1/6 \approx 0.167$ ở trạng thái `"initialized"`. Khi có lidar hit, score tăng thêm $1/\text{window}$ (tối đa bằng 1.0) và chuyển sang `"tentative"`. Khi $\text{score} > \text{confirmed\_threshold}$ ($0.8$), track được thăng hạng thành `"confirmed"`.
   - **Giữ confirmed sau miss:** Khi track đã đạt trạng thái `"confirmed"`, nếu gặp lidar miss trong FOV, điểm score bị trừ $1/\text{window}$ nhưng **trạng thái vẫn được giữ nguyên là `"confirmed"`**, không bị hạ cấp về `"tentative"` hay `"initialized"`.
   - **Điều kiện xóa track (`should_delete_track`):** Track bị xóa khỏi hệ thống khi thỏa mãn bất kỳ điều kiện nào sau đây (logic OR):
     1. Hiệp phương sai vị trí ngang quá lớn: $P[0, 0] > \text{max\_P}$ hoặc $P[1, 1] > \text{max\_P}$ (với $\text{max\_P} = 3.0^2 = 9.0 \, \text{m}^2$).
     2. Track đã `"confirmed"` nhưng bị miss nhiều lần khiến $\text{score} < \text{delete\_threshold}$ ($0.6$).
     3. Track chưa `"confirmed"` (trạng thái `"initialized"` hoặc `"tentative"`) có $\text{score} \le 0.0$.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Assistant
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích công thức toán ma trận EKF (F, Q, S, K), viết các hàm trong workspace (kalman, camera_fusion, association, track_management) và rà soát các câu trả lời lý thuyết.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Tự chạy kiểm thử với `pytest student/tests` đạt 128/128 test pass (100%), chạy đánh giá pipeline Waymo với `fusion-run-lab` trên 199 frames, kiểm tra tính toán sai số RMSE khớp hoàn toàn với `grade_run.log` và xác minh bằng script `tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
