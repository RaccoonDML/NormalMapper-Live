import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


def generate_shadow(
    normal_path,
    light_position=(1000, -1000, 1000),  
    threshold=0.3,
    smoothing=0,
    ind=None,
):  
    
    if isinstance(normal_path, str):
        normal_map = cv2.imread(normal_path)
    elif isinstance(normal_path, Image.Image):
        normal_map = np.array(normal_path)
        # RGB to BGR
        normal_map = normal_map[..., ::-1]
    else:
        print(f"Input is numpy array with shape: {normal_path.shape}, dtype: {normal_path.dtype}")
        normal_map = normal_path.copy()
    
    if normal_map is None:
        raise ValueError("Failed to load or convert normal map")
        
    
    normal_map = normal_map.astype(np.float32) / 127.5 - 1.0
    
    height, width = normal_map.shape[:2]
    x_pos, y_pos, z_pos = light_position

    background_mask = np.all(np.abs(normal_map - np.array([1.0, 1.0, 1.0])) < 0.1, axis=2)
    print(f"Background mask covers {np.sum(background_mask)/(width*height)*100:.1f}% of image")
    
    if ind is None:
        # 生成行、列索引网格
        rows, cols = np.indices((height, width))
        # 直接构建三维坐标网格
        ind = np.stack([
            np.zeros_like(rows),  # Z轴（初始为0）
            rows,                 # Y轴（行号）
            cols                  # X轴（列号）
        ], axis=-1).astype(np.float32)
    
    if smoothing > 0:
        eps = 0.04
        I = normal_map
        I2 = cv2.pow(I,2)
        mean_I = cv2.boxFilter(I,-1,((2*smoothing)+1,(2*smoothing)+1))
        mean_I2 = cv2.boxFilter(I2,-1,((2*smoothing)+1,(2*smoothing)+1))
        cov_I = mean_I2 - cv2.pow(mean_I,2)
        var_I = cov_I
        a = cv2.divide(cov_I,var_I+eps)
        b = mean_I - (a*mean_I)
        mean_a = cv2.boxFilter(a,-1,((2*smoothing)+1,(2*smoothing)+1))
        mean_b = cv2.boxFilter(b,-1,((2*smoothing)+1,(2*smoothing)+1))
        normal_map = (mean_a * I) + mean_b

    Z = ind[:,:,0] + z_pos
    Y = ind[:,:,1] + y_pos
    X = ind[:,:,2] - x_pos
    SUM = np.sqrt(X**2 + Y**2 + Z**2)
    LD = np.zeros_like(ind)
    LD[:,:,0] = Z / SUM
    LD[:,:,1] = Y / SUM
    LD[:,:,2] = X / SUM

    dot = np.sum(normal_map * LD, axis=2)
    dot = np.clip(dot, 0, 1.0)

    if threshold > 0:
        shadow_map = np.where(dot > threshold, 0, 255).astype(np.uint8)
    else:
        shadow_map = ((1-dot)*255).astype(np.uint8)

    shadow_map[background_mask] = 0  # 背景区域设为白色
    
    result = Image.fromarray(255-shadow_map)
    return result



class ShadowGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.normal_map = None
        self.ind = None
        self.xy_pos = None

    def initUI(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 控制面板
        control_panel = QGroupBox("参数控制")
        control_layout = QFormLayout()
        
        # 文件选择
        self.file_btn = QPushButton("选择法线贴图")
        self.file_label = QLabel("未选择文件")
        control_layout.addRow(self.file_btn, self.file_label)
        
        # 添加保存按钮
        self.save_btn = QPushButton("保存阴影图")
        self.save_btn.setEnabled(False)  # 初始时禁用
        control_layout.addRow(self.save_btn)
        
        # 光源位置
        x_container, self.x_slider = self.create_slider(-500, 1500, 500)
        y_container, self.y_slider = self.create_slider(-1500, 500, -500)
        z_container, self.z_slider = self.create_slider(-2000, 2000, 1000)
        control_layout.addRow("X 位置:", x_container)
        control_layout.addRow("Y 位置:", y_container)
        control_layout.addRow("Z 位置:", z_container)

        # 阈值
        threshold_container, self.threshold_slider = self.create_slider(0, 10, 5)
        control_layout.addRow("阈值:", threshold_container)

        # 平滑
        smooth_container, self.smooth_slider = self.create_slider(0, 5, 3)
        control_layout.addRow("平滑:", smooth_container)
        
        # Set the layout to the control panel
        control_panel.setLayout(control_layout)

        # 图像显示
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建图像容器
        image_container = QWidget()
        self.image_container_layout = QHBoxLayout(image_container)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 设置标签
        self.normal_label = QLabel()
        self.normal_label.setMinimumSize(200, 200)
        self.normal_label.setAlignment(Qt.AlignCenter)
        self.shadow_label = QLabel()
        self.shadow_label.setMinimumSize(200, 200)
        self.shadow_label.setAlignment(Qt.AlignCenter)
        
        self.image_container_layout.addWidget(self.normal_label)
        self.image_container_layout.addWidget(self.shadow_label)
        
        scroll_area.setWidget(image_container)
        image_layout.addWidget(scroll_area)
        
        # 组合布局
        layout.addWidget(control_panel, 1)
        layout.addWidget(image_panel, 2)

        # 信号连接
        self.file_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_shadow)
        for slider in [self.x_slider, self.y_slider, self.z_slider,
                      self.threshold_slider, self.smooth_slider]:
            slider.valueChanged.connect(self.update_shadow)

        # 窗口设置
        self.setWindowTitle("实时阴影生成器")
        self.resize(1200, 600)

    def create_slider(self, min_val, max_val, default):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(50)
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        # Update label when slider value changes
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        return container, slider

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择法线贴图", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.normal_map = cv2.imread(path)
                if self.normal_map is None:
                    QMessageBox.critical(self, "错误", "无法加载图像文件。请确保文件格式正确。")
                    return
                self.file_label.setText(path.split("/")[-1])
                self.show_normal_map()
                self.update_shadow()

                rows, cols = np.indices((self.normal_map.shape[0], self.normal_map.shape[1]))
                # 直接构建三维坐标网格
                self.ind = np.stack([
                    np.zeros_like(rows),  # Z轴（初始为0）
                    rows,                 # Y轴（行号）
                    cols                  # X轴（列号）
                ], axis=-1).astype(np.float32)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像时出错：{str(e)}")

    def show_normal_map(self):
        if self.normal_map is not None:
            rgb = cv2.cvtColor(self.normal_map, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, QImage.Format_RGB888)
            self.normal_pixmap = QPixmap.fromImage(qimg)
            self.update_image_display()

    def save_shadow(self):
        try:
            if not hasattr(self, 'current_shadow'):
                QMessageBox.warning(self, "警告", "没有可保存的阴影图")
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存阴影图",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*.*)"
            )
            
            if file_path:
                # 确保文件有正确的扩展名
                if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    file_path += '.png'
                
                self.current_shadow.save(file_path)
                QMessageBox.information(self, "成功", "阴影图保存成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件时出错：{str(e)}")

    def update_shadow(self):
        if self.normal_map is None:
            return

        try:
            # 获取参数
            params = {
                'light_position': (
                    self.x_slider.value(),
                    self.y_slider.value(),
                    self.z_slider.value()
                ),
                'threshold': self.threshold_slider.value()/10,
                'smoothing': self.smooth_slider.value()
            }
            print(f"\nProcessing with parameters: {params}")

            # 生成阴影
            shadow_img = generate_shadow(self.normal_map, **params, ind=self.ind)
            
            # 保存当前的阴影图用于导出
            self.current_shadow = shadow_img
            # 启用保存按钮
            self.save_btn.setEnabled(True)
            
            # 显示结果
            shadow_np = np.array(shadow_img)
            qimg = QImage(shadow_np.data, shadow_np.shape[1], shadow_np.shape[0], 
                         QImage.Format_Grayscale8)
            self.shadow_pixmap = QPixmap.fromImage(qimg)
            self.update_image_display()
            
        except Exception as e:
            import traceback
            print(f"Error in update_shadow: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.warning(self, "警告", f"生成阴影时出错：{str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

    def update_image_display(self):
        if not hasattr(self, 'normal_map'):
            return
            
        # 获取可用空间
        container_width = self.image_container_layout.contentsRect().width() // 2
        container_height = self.image_container_layout.contentsRect().height()
        
        if container_width <= 0 or container_height <= 0:
            return
            
        # 更新法线图显示
        if hasattr(self, 'normal_pixmap'):
            scaled_normal = self.normal_pixmap.scaled(
                container_width, container_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.normal_label.setPixmap(scaled_normal)
            
        # 更新阴影图显示
        if hasattr(self, 'shadow_pixmap'):
            scaled_shadow = self.shadow_pixmap.scaled(
                container_width, container_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.shadow_label.setPixmap(scaled_shadow)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShadowGUI()
    window.show()
    sys.exit(app.exec_())