import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

is_playing = False
marker_size = []
tail = 10
start_date, end_date = None, None

for i in range(tail):
    if i == tail-1:
        marker_size.append(50)
    else:
        marker_size.append(10)

def get_line_points(x, y):
    # Interpolate a smooth curve through the scatter points
    tck, _ = interpolate.splprep([x, y], s=0)
    t = np.linspace(0, 1, 100)
    line_x, line_y = interpolate.splev(t, tck)
    return line_x, line_y

def get_status(x, y):
    if x < 100 and y < 100:
        return 'lagging'
    elif x > 100 and y > 100:
        return 'leading'
    elif x < 100 and y > 100:
        return 'improving'
    elif x > 100 and y < 100:
        return 'weakening'
    
def get_color(x, y):
    if get_status(x, y) == 'lagging':
        return 'red'
    elif get_status(x, y) == 'leading':
        return 'green'
    elif get_status(x, y) == 'improving':
        return 'blue'
    elif get_status(x, y) == 'weakening':
        return 'yellow'

def load_csv_data(filename):
    """Load CSV data and return a pandas Series with datetime index and close prices"""
    try:
        df = pd.read_csv(filename)
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        # Return the close prices as a Series
        return df['close']
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Configuration - Add your ticker CSV files here
# Format: {'display_name': 'csv_filename.csv', ...}
TICKER_FILES = {
    'S99':'price_data/S99_1D.csv'
}

# Load benchmark data
benchmark_data = load_csv_data('price_data/VNINDEX_1D.csv')
if benchmark_data is None:
    print("Error: Could not load benchmark data from VNINDEX_1D.csv")
    exit()

# Load ticker data
tickers = list(TICKER_FILES.keys())
tickers_data = pd.DataFrame()

for ticker, filename in TICKER_FILES.items():
    data = load_csv_data(filename)
    if data is not None:
        tickers_data[ticker] = data
    else:
        print(f"Warning: Could not load data for {ticker} from {filename}")
        tickers.remove(ticker)

# Align all data to have the same dates
all_data = pd.concat([benchmark_data, tickers_data], axis=1)
all_data = all_data.dropna()  # Remove rows with any NaN values
benchmark_data = all_data.iloc[:, 0]
tickers_data = all_data.iloc[:, 1:]

# Update tickers list to match loaded data
tickers = list(tickers_data.columns)
tickers_to_show = tickers.copy()

# Calculate RRG indicators
window = 14

rs_tickers = []
rsr_tickers = []
rsr_roc_tickers = []
rsm_tickers = []

for ticker in tickers:
    rs = 100 * (tickers_data[ticker] / benchmark_data)
    rs_tickers.append(rs)
    
    rsr = (100 + (rs - rs.rolling(window=window).mean()) / rs.rolling(window=window).std(ddof=0)).dropna()
    rsr_tickers.append(rsr)
    
    rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
    rsr_roc_tickers.append(rsr_roc)
    
    rsm = (101 + ((rsr_roc - rsr_roc.rolling(window=window).mean()) / rsr_roc.rolling(window=window).std(ddof=0))).dropna()
    rsm_tickers.append(rsm)
    
    # Align indices
    common_index = rsr.index.intersection(rsm.index)
    idx = tickers.index(ticker)
    rsr_tickers[idx] = rsr_tickers[idx].loc[common_index]
    rsm_tickers[idx] = rsm_tickers[idx].loc[common_index]

def update_rrg():
    global rs_tickers, rsr_tickers, rsr_roc_tickers, rsm_tickers
    rs_tickers = []
    rsr_tickers = []
    rsr_roc_tickers = []
    rsm_tickers = []

    for ticker in tickers:
        rs = 100 * (tickers_data[ticker] / benchmark_data)
        rs_tickers.append(rs)
        
        rsr = (100 + (rs - rs.rolling(window=window).mean()) / rs.rolling(window=window).std(ddof=0)).dropna()
        rsr_tickers.append(rsr)
        
        rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
        rsr_roc_tickers.append(rsr_roc)
        
        rsm = (101 + ((rsr_roc - rsr_roc.rolling(window=window).mean()) / rsr_roc.rolling(window=window).std(ddof=0))).dropna()
        rsm_tickers.append(rsm)
        
        # Align indices
        common_index = rsr.index.intersection(rsm.index)
        idx = tickers.index(ticker)
        rsr_tickers[idx] = rsr_tickers[idx].loc[common_index]
        rsm_tickers[idx] = rsm_tickers[idx].loc[common_index]

# Create GUI
root = tk.Tk()
root.title('RRG Indicator - CSV Data')
root.geometry('1000x650')
root.resizable(False, False)

# Create scatter plot
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax[0].set_title('RRG Indicator')
ax[0].set_xlabel('JdK RS Ratio')
ax[0].set_ylabel('JdK RS Momentum')

# Add horizontal and vertical lines to (100, 100) origin 
ax[0].axhline(y=100, color='k', linestyle='--')
ax[0].axvline(x=100, color='k', linestyle='--')

# Color each quadrant
ax[0].fill_between([94, 100], [94, 94], [100, 100], color='red', alpha=0.2)
ax[0].fill_between([100, 106], [94, 94], [100, 100], color='yellow', alpha=0.2)
ax[0].fill_between([100, 106], [100, 100], [106, 106], color='green', alpha=0.2)
ax[0].fill_between([94, 100], [100, 100], [106, 106], color='blue', alpha=0.2)

# Add text labels in each corner
ax[0].text(95, 105, 'Improving')
ax[0].text(104, 105, 'Leading')
ax[0].text(104, 95, 'Weakening')
ax[0].text(95, 95, 'Lagging')

ax[0].set_xlim(94, 106)
ax[0].set_ylim(94, 106)

# Add plot to canvas 
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

ax[1].set_axis_off()

# Add a slider for the end date 
ax_end_date = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='grey')
slider_end_date = Slider(ax_end_date, 'Date', tail, len(rsr_tickers[0])-2, valinit=tail, valstep=1, initcolor='none', track_color='grey')
slider_end_date.poly.set_fc('grey')
date = str(rsr_tickers[0].index[slider_end_date.val]).split(' ')[0]
slider_end_date.valtext.set_text(date)

def update_slider_end_date(val):
    date = str(rsr_tickers[0].index[val]).split(' ')[0]
    slider_end_date.valtext.set_text(date)

slider_end_date.on_changed(update_slider_end_date)

# Get the real date from the slider value
start_date = rsr_tickers[0].index[0]
end_date = rsr_tickers[0].index[slider_end_date.val]

# Add a slider for the tail 
ax_tail = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_tail = Slider(ax_tail, 'Tail', 1, 10, valinit=5, valstep=1, initcolor='none', track_color='grey')
slider_tail.poly.set_fc('grey')

def update_slider_tail(val):
    global tail
    global marker_size
    # Check if the end date - tail is less than the start date 
    if slider_end_date.val - slider_tail.val < slider_end_date.valmin:
        slider_tail.eventson = False
        slider_tail.set_val(tail)
        slider_tail.eventson = True
        return
    # Update the min of the end date slider 
    slider_end_date.valmin = slider_tail.val
    slider_end_date.ax.set_xlim(slider_tail.val, slider_end_date.valmax)
    tail = slider_tail.val
    marker_size = []
    for i in range(tail):
        if i == tail-1:
            marker_size.append(50)
        else:
            marker_size.append(10)

slider_tail.on_changed(update_slider_tail)

# Add a button to play the animation 
ax_play = plt.axes([0.05, 0.02, 0.1, 0.04])
button_play = Button(ax_play, 'Play')

def update_button_play(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        button_play.label.set_text('Pause')
    else:
        button_play.label.set_text('Play')

button_play.on_clicked(update_button_play)

# Create table
table = tk.Frame(master=root)
table.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

headers = ['Symbol', 'Price', 'Change %', 'Status', 'Visible']
widths = [15, 15, 15, 20, 10]
for j in range(len(headers)):
    tk.Label(table, text=headers[j], relief=tk.RIDGE, width=widths[j], font=('Arial', 12, 'bold')).grid(row=0, column=j)

def update_check_button(event):
    global tickers_to_show
    
    check_button = event.widget
    row = check_button.grid_info()['row']
    # Get ticker symbol from the table 
    symbol = tickers[row-1]
    
    # If the check button is checked, add the ticker to the list of tickers to show
    if 'selected' not in check_button.state() and symbol not in tickers_to_show:
        tickers_to_show.append(symbol)
    elif 'selected' in check_button.state() and symbol in tickers_to_show:
        tickers_to_show = [x for x in tickers_to_show if x != symbol]

# Populate table
for i, ticker in enumerate(tickers):
    # Ticker price at end date
    price = round(tickers_data[ticker].loc[end_date], 2)
    # Ticker change from start date to end date in percentage
    chg = round((price - tickers_data[ticker].loc[start_date]) / tickers_data[ticker].loc[start_date] * 100, 1)
    # Get status and color
    status = get_status(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
    bg_color = get_color(rsr_tickers[i].iloc[-1], rsm_tickers[i].iloc[-1])
    fg_color = 'white' if bg_color in ['red', 'green'] else 'black'
    
    tk.Label(table, text=ticker, relief=tk.RIDGE, width=15, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(row=i+1, column=0)
    tk.Label(table, text=price, relief=tk.RIDGE, width=15, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(row=i+1, column=1)
    tk.Label(table, text=f"{chg}%", relief=tk.RIDGE, width=15, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(row=i+1, column=2)
    tk.Label(table, text=status.capitalize(), relief=tk.RIDGE, width=20, bg=bg_color, fg=fg_color, font=('Arial', 12)).grid(row=i+1, column=3)
    
    checkbox_var = tk.BooleanVar()
    checkbox_var.set(True)
    checkbox = ttk.Checkbutton(table, variable=checkbox_var, onvalue=True, offvalue=False)
    checkbox.grid(row=i+1, column=4)
    checkbox.state(['selected'])
    checkbox.bind('<Button-1>', update_check_button)

# Animation setup
scatter_plots = [] 
line_plots = []
annotations = []

for i in range(len(tickers)):
    scatter_plots.append(ax[0].scatter([], []))
    line_plots.append(ax[0].plot([], [], color='k', alpha=0.2)[0]) 
    annotations.append(ax[0].annotate(tickers[i], (0, 0), fontsize=8))

# Animation function
def animate(frame):
    global start_date, end_date

    if not is_playing:
        # Take the value from the slider 
        end_date = rsr_tickers[0].index[slider_end_date.val]
        start_date = rsr_tickers[0].index[slider_end_date.val - tail]
    else:
        # Move forward one week
        current_idx = rsr_tickers[0].index.get_loc(end_date)
        if current_idx < len(rsr_tickers[0].index) - 1:
            start_date = rsr_tickers[0].index[current_idx - tail + 1]
            end_date = rsr_tickers[0].index[current_idx + 1]
            # Update slider
            slider_end_date.eventson = False
            slider_end_date.set_val(current_idx + 1)
            slider_end_date.eventson = True
        else:
            # Reset to beginning
            start_date = rsr_tickers[0].index[0]
            end_date = rsr_tickers[0].index[tail]
            slider_end_date.eventson = False
            slider_end_date.set_val(tail)
            slider_end_date.eventson = True

    for j in range(len(tickers)):
        # If ticker not to be displayed, clear it
        if tickers[j] not in tickers_to_show:
            scatter_plots[j] = ax[0].scatter([], [])
            line_plots[j] = ax[0].plot([], [], color='k', alpha=0.2)[0]
            annotations[j] = ax[0].annotate('', (0, 0), fontsize=8)
        else:
            # Filter data for current window
            filtered_rsr = rsr_tickers[j].loc[(rsr_tickers[j].index > start_date) & (rsr_tickers[j].index <= end_date)]
            filtered_rsm = rsm_tickers[j].loc[(rsm_tickers[j].index > start_date) & (rsm_tickers[j].index <= end_date)]
            
            if len(filtered_rsr) > 0:
                # Update scatter plot
                color = get_color(filtered_rsr.values[-1], filtered_rsm.values[-1])
                scatter_plots[j] = ax[0].scatter(filtered_rsr.values, filtered_rsm.values, color=color, s=marker_size[:len(filtered_rsr)])
                # Update line
                line_plots[j] = ax[0].plot(filtered_rsr.values, filtered_rsm.values, color='black', alpha=0.2)[0]
                # Update annotation
                annotations[j] = ax[0].annotate(tickers[j], (filtered_rsr.values[-1], filtered_rsm.values[-1]))
                
                # Update table
                price = round(tickers_data[tickers[j]].loc[end_date], 2)
                chg = round((price - tickers_data[tickers[j]].loc[start_date]) / tickers_data[tickers[j]].loc[start_date] * 100, 1)
                status = get_status(filtered_rsr.values[-1], filtered_rsm.values[-1])
                bg_color = get_color(filtered_rsr.values[-1], filtered_rsm.values[-1])
                fg_color = 'white' if bg_color in ['red', 'green', 'blue'] else 'black'
                
                table.grid_slaves(row=j+1, column=1)[0].config(text=price, bg=bg_color, fg=fg_color)
                table.grid_slaves(row=j+1, column=2)[0].config(text=f"{chg}%", bg=bg_color, fg=fg_color)
                table.grid_slaves(row=j+1, column=3)[0].config(text=status.capitalize(), bg=bg_color, fg=fg_color)
                table.grid_slaves(row=j+1, column=0)[0].config(bg=bg_color, fg=fg_color)

    return scatter_plots + line_plots + annotations

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=60, interval=100, blit=True)

root.mainloop()