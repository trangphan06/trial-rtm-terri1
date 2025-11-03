import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io  # Cần thiết để tạo file download trong bộ nhớ

def run_territory_planning(input_file, n_clusters, min_size=None, max_size=None, n_init=10):
    """
    Chạy K-Means clustering với ràng buộc min/max.
    Đã được chỉnh sửa cho Streamlit.
    """

    # --- 1. Tải Dữ liệu ---
    st.info(f"Đang tải dữ liệu khách hàng...")
    try:
        # 'input_file' giờ là một đối tượng file được upload, không phải đường dẫn
        df = pd.read_excel(input_file)
    except Exception as e:
        st.error(f"Lỗi khi đọc file Excel: {e}")
        return None, None

    if 'lat' not in df.columns or 'long' not in df.columns:
        st.error("Lỗi: File Excel phải chứa cột 'lat' và 'long'.")
        return None, None

    df_original = df.copy()
    coords = df[['lat', 'long']]
    st.info(f"Đã tải {len(coords)} khách hàng.")

    # --- 2. Chuẩn bị Dữ liệu ---
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # --- 3. Thiết lập Ràng buộc ---
    if min_size is None or min_size == 0:
        min_size = 1
    if max_size is None or max_size == 0:
        max_size = len(df)

    if min_size * n_clusters > len(df):
        st.error(f"Lỗi: Ràng buộc TỐI THIỂU không thể thực hiện.")
        st.error(f"({min_size} min * {n_clusters} clusters = {min_size * n_clusters} KH yêu cầu)")
        st.error(f"Bạn chỉ có {len(df)} tổng số khách hàng.")
        return None, None

    if max_size * n_clusters < len(df):
        st.error(f"Lỗi: Ràng buộc TỐI ĐA không thể thực hiện.")
        st.error(f"({max_size} max * {n_clusters} clusters = {max_size * n_clusters} KH tối đa)")
        st.error(f"Ít hơn {len(df)} tổng số khách hàng của bạn.")
        return None, None

    # --- 4. Chạy Constrained K-Means ---
    st.info(f"Đang chạy {n_init} lần khởi tạo cho {n_clusters} tuyến...")
    st.info(f"Ràng buộc: Min={min_size}, Max={max_size} KH mỗi tuyến.")

    best_model = None
    best_inertia = np.inf
    
    # Thay thế tqdm bằng st.progress
    progress_bar = st.progress(0, text="Đang chạy phân cụm...")

    try:
        for i in range(n_init):
            model = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=min_size,
                size_max=max_size,
                random_state=42 + i,
                n_init=1
            )
            model.fit(coords_scaled)

            if model.inertia_ < best_inertia:
                best_inertia = model.inertia_
                best_model = model
            
            # Cập nhật thanh tiến trình
            progress_bar.progress((i + 1) / n_init, text=f"Đang chạy lần {i + 1}/{n_init}")

    except Exception as e:
        progress_bar.empty()
        st.error("--- PHÂN CỤM THẤT BẠI ---")
        st.error("Điều này có thể xảy ra nếu ràng buộc (min/max/số cụm) quá khó.")
        st.error(f"Chi tiết lỗi: {e}")
        return None, None

    progress_bar.empty() # Xóa thanh tiến trình khi hoàn tất

    if best_model is None:
        st.error("Lỗi: Không thể hoàn tất phân cụm.")
        return None, None

    # --- 5. Trả về Kết quả ---
    df_original['territory_id'] = best_model.labels_
    
    # Tạo báo cáo
    report_lines = []
    report_lines.append("--- Phân tuyến thành công! ---")
    report_lines.append("\nBáo cáo phân tuyến (SR Workload):")
    cluster_counts = df_original['territory_id'].value_counts().sort_index()
    cluster_counts.index.name = 'Territory ID'
    
    # Chuyển DataFrame báo cáo thành chuỗi
    report_lines.append(cluster_counts.to_string()) 
    report_lines.append(f"\nTổng số KH được gán: {cluster_counts.sum()}")
    report_lines.append(f"Trung bình mỗi tuyến: {cluster_counts.mean():.1f}")
    
    report_string = "\n".join(report_lines)

    return df_original, report_string

# =============================================================================
# GIAO DIỆN WEB (STREAMLIT UI)
# =============================================================================

st.set_page_config(layout="wide")
st.title("Công cụ lập kế hoạch phân tuyến (Territory Planning)")

# --- Thanh bên (Sidebar) để chứa các nút điều khiển ---
with st.sidebar:
    st.header("Cài đặt đầu vào")

    # 1. Widget Upload File
    uploaded_file = st.file_uploader("Tải lên file Excel (.xlsx, .xls)", type=['xlsx', 'xls'])
    
    st.markdown("---")
    
    # 2. Widgets cho các tham số
    n_routes = st.number_input(
        "Số lượng tuyến (n_clusters)", 
        min_value=1, 
        value=9,  # Giá trị mặc định
        step=1
    )
    
    min_customers = st.number_input(
        "Số KH TỐI THIỂU mỗi tuyến (min_size)", 
        min_value=0, 
        value=260, # Giá trị mặc định
        step=1
    )
    
    max_customers = st.number_input(
        "Số KH TỐI ĐA mỗi tuyến (max_size)", 
        min_value=0, 
        value=330, # Giá trị mặc định
        step=1
    )
    
    n_init_runs = st.number_input(
        "Số lần chạy (n_init - càng cao càng tốt nhưng chậm hơn)", 
        min_value=1, 
        value=50,  # Giá trị mặc định
        step=10
    )

    # 3. Nút chạy
    run_button = st.button("Bắt đầu phân tuyến")

# --- Khu vực hiển thị chính ---
if run_button:
    if uploaded_file is not None:
        # Gọi hàm logic chính
        df_result, report_text = run_territory_planning(
            input_file=uploaded_file,
            n_clusters=n_routes,
            min_size=min_customers,
            max_size=max_customers,
            n_init=n_init_runs
        )
        
        # Nếu hàm chạy thành công
        if df_result is not None:
            st.success("Hoàn tất! Kết quả ở bên dưới.")
            
            # Hiển thị báo cáo
            st.subheader("Báo cáo kết quả")
            st.text(report_text) # Hiển thị báo cáo dạng text
            
            # Hiển thị xem trước DataFrame
            st.subheader("Xem trước dữ liệu kết quả")
            st.dataframe(df_result.head(10))
            
            # --- Tạo nút Download ---
            output_buffer = io.BytesIO()
            with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Territory_Output')
            
            st.download_button(
                label="Tải file Excel kết quả",
                data=output_buffer.getvalue(),
                file_name="territory_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.warning("Vui lòng tải lên một file Excel trước khi chạy.")
else:
    st.info("Vui lòng thiết lập các tham số ở thanh bên (sidebar) và nhấn 'Bắt đầu phân tuyến'.")
