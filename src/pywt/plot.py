import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from pywt.wavetable import Wavetable


def view_wavetable(wavetable: Wavetable, /):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(left=0.1, bottom=0.25)
    frames = wavetable.frames()
    frame_idx_init = 0
    frame_init = frames[frame_idx_init]

    # time domain data
    _, frame_init_time_domain_data = frame_init.plot_time_domain()
    line_time_domain, = axs[0][0].plot(np.arange(wavetable.frame_size), frame_init_time_domain_data, lw=2, color='red',
                                       label='x(t)')

    # FFT domain data
    frame_init_fft_domain_amp, frame_init_time_domain_phase = frame_init.plot_freq_domain()
    line_fft_amp, = axs[0][1].plot(np.arange(wavetable.n_partials), frame_init_fft_domain_amp, lw=2, color='blue',
                                   label='X(f)', marker='.', linestyle='')
    axs[0][1].set_yscale('log')
    line_fft_phase, = axs[1][1].plot(np.arange(wavetable.n_partials), frame_init_time_domain_phase, lw=2,
                                     color='blue',
                                     label=r'\phi(f)', marker='.', linestyle='')

    reset_line_color = 'black'
    ax_frame = plt.axes((0.1, 0.1, 0.65, 0.03), facecolor=reset_line_color)

    slider_frame = Slider(ax_frame, 'Frame', 0, wavetable.number_of_frames, valinit=frame_idx_init,
                          valstep=np.arange(wavetable.number_of_frames))

    def update(val):
        frame_idx = slider_frame.val
        line_time_domain.set_ydata(frames[frame_idx].plot_time_domain()[1])
        amp, phase = frames[frame_idx].plot_freq_domain()
        line_fft_amp.set_ydata(amp)
        line_fft_phase.set_ydata(phase)
        fig.canvas.draw_idle()

    slider_frame.on_changed(update)

    ax_reset = plt.axes((0.8, 0.025, 0.1, 0.04))
    button = Button(ax_reset, 'Reset', color='white', hovercolor='0.975')

    def reset(event):
        slider_frame.reset()

    button.on_clicked(reset)

    plt.show()
