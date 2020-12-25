
## ---

### Rate calculation used in mingpt
function cosRate( tokens, lr=6e-4, warmuptoken=512*20, len_train_dataset=1115394, blocksize=128)
    finalTok=2*len_train_dataset*blocksize
    progress = (tokens - warmuptoken) / max(1, finalTok - warmuptoken)
    lr_mult = max(0.1, 0.5 * (1.0 + cos(pi * progress)))
    lr_mult*lr
end

@show cosRate(60*512*128)
@show cosRate(125*512*128)
@show cosRate(235*512*128)
@show cosRate(350*512*128)
