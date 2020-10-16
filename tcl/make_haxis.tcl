proc read_all_pdb_files {start end} {
    mol new haxis.$start.pdb type pdb
    for {set index [expr $start + 1]} { $index < [expr $end + 1] } { incr index } {
        mol addfile haxis.$index.pdb 0
    }
}