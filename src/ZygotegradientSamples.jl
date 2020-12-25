using Zygote

begin
W = rand(2, 3); x = rand(3);
@show gradient(W -> sum(W*x), W)
# Since we only have W as parameter, the first element is the grad
# a 2x3 array
@show gradient(W -> sum(W*x), W)[1]
end


begin
d = Dict()
# Note the do is a x-> ....  function
# Gradient of that function is calculated at x=5
# function acts like x**2  so  its gradient is 2x=2*5=10
@show  gradient(5) do x
         d[:x] = x
         d[:x] * d[:x]
       end


@show  d[:x]
end
