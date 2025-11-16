#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
import pickle
import os

# === PARAMETERS ===
SEGMENT_SIZE = 4   # number of points per segment
LABELS = ['flat', 'spiky', 'tight']
COLORS = np.array([
    [0.5, 0.5, 0.5],  # flat - gray
    [1.0, 0.0, 0.0],  # spiky - red
    [0.0, 0.3, 1.0],  # tight - blue
])
AUTO_SAVE = False    # save automatically after each change
SAVE_PATH = "labels_multilabel.pkl"

# === LOAD YOUR DATA HERE ===
theta = np.linspace(0, 6*np.pi, 300)
r = np.linspace(0, 1, 300) + 0.1*np.sin(10*theta)
x, y = r * np.cos(theta), r * np.sin(theta)

# === LABEL STORAGE ===
num_points = len(r)
labels = np.zeros((num_points, len(LABELS)), dtype=int)

# === COLOR MIXING ===
def mix_colors(label_matrix):
    mixed = label_matrix @ COLORS
    mixed = np.clip(mixed, 0, 1)
    mixed[label_matrix.sum(axis=1) == 0] = [0.8, 0.8, 0.8]
    return mixed

# === SETUP PLOT ===
fig, ax = plt.subplots(figsize=(7,7))
plt.subplots_adjust(left=0.3, bottom=0.25)
ax.set_title("Spiral Multi-Label Tool (Keyboard + Segments)")
ax.axis("equal")

sc = ax.scatter(x, y, c=mix_colors(labels), s=20)
highlight, = ax.plot([], [], "ko", markersize=12, fillstyle="none")

# === SLIDER ===
num_segments = int(np.ceil(num_points / SEGMENT_SIZE))
ax_slider = plt.axes([0.35, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Segment", 0, num_segments-1, valinit=0, valstep=1)

# === CHECKBOXES ===
rax = plt.axes([0.05, 0.5, 0.2, 0.15])
check = CheckButtons(rax, LABELS, [False]*len(LABELS))

# === SAVE BUTTON ===
save_ax = plt.axes([0.08, 0.35, 0.1, 0.05])
save_button = Button(save_ax, "Save")

# === LOGIC ===
def save_labels():
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(labels, f)
    print(f"💾 Labels saved to {os.path.abspath(SAVE_PATH)}")

def update_highlight(segment):
    seg = int(segment)
    start = seg * SEGMENT_SIZE
    end = min(start + SEGMENT_SIZE, num_points)
    highlight.set_data(x[start:end], y[start:end])
    # Sync checkboxes with current segment
    for i in range(len(LABELS)):
        seg_label = labels[start:end, i].mean() > 0.5
        if seg_label != check.get_status()[i]:
            check.set_active(i)
    fig.canvas.draw_idle()

def toggle_label(i):
    seg = int(slider.val)
    start = seg * SEGMENT_SIZE
    end = min(start + SEGMENT_SIZE, num_points)
    labels[start:end, i] = 1 - labels[start:end, i]
    sc.set_color(mix_colors(labels))
    update_highlight(slider.val)
    if AUTO_SAVE:
        save_labels()

def clear_labels():
    seg = int(slider.val)
    start = seg * SEGMENT_SIZE
    end = min(start + SEGMENT_SIZE, num_points)
    labels[start:end, :] = 0
    sc.set_color(mix_colors(labels))
    update_highlight(slider.val)
    if AUTO_SAVE:
        save_labels()

def move_segment(offset):
    new_val = np.clip(slider.val + offset, 0, num_segments-1)
    slider.set_val(new_val)
    update_highlight(new_val)

# === EVENTS ===
def on_key(event):
    if event.key == "left":
        move_segment(-1)
    elif event.key == "right":
        move_segment(1)
    elif event.key == "f":
        toggle_label(0)
    elif event.key == "s":
        toggle_label(1)
    elif event.key == "t":
        toggle_label(2)
    elif event.key == " ":
        clear_labels()
    elif event.key == "ctrl+s":
        save_labels()
    fig.canvas.draw_idle()

def on_check(label_name):
    i = LABELS.index(label_name)
    toggle_label(i)

def on_save(event):
    save_labels()

# === WIRING ===
slider.on_changed(update_highlight)
check.on_clicked(on_check)
save_button.on_clicked(on_save)
fig.canvas.mpl_connect("key_press_event", on_key)

update_highlight(0)
plt.show()