proc vmd_draw_arrow {mol start end} {
    # an arrow is made of a cylinder and a cone
    set middle [vecadd $start [vecscale 0.6 [vecsub $end $start]]]
    graphics $mol cylinder $start $middle radius 0.25
    graphics $mol cone $middle $end radius 0.45
}