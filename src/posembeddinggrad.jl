
begin
	using Functors:functor
	using Flux

	using Transformers
	using Transformers.Basic #for loading the positional embedding
	using Flux: @functor, Params
	using Flux: update!

	#flux function for update parameters
	using Flux: gradient
	using Statistics: norm, mean, std

	labels = collect(1:10)
	startsym = 11
	endsym = 12
	unksym = 0
	labels = [unksym, startsym, endsym, labels...]
	vocab = Vocabulary(labels, unksym)
	#function for generate training datas
	sample_data() = rand(1:10, 10)
	#function for adding start & end symbol
	preprocess(x) = [startsym, x..., endsym]

	sample = preprocess(sample_data())

	batch=hcat(preprocess.((sample_data() for i = 1:32))...)
end


begin
	mutable struct  simpleLayer
		embd::Embed
		pe::PositionEmbedding
		linear::Positionwise
	end

	@functor simpleLayer

	function simpleLayer()
		embd=Embed(512, length(vocab))
		pe = PositionEmbedding(512, trainable=true)
		linear = Positionwise(Dense(512, length(vocab)))
		simpleLayer(embd, pe, linear)
	end

	function (t::simpleLayer)(x)
		we = t.embd(x, inv(sqrt(512)))
		e = we .+ t.pe(we)
		p = t.linear(e)
	end

	model=simpleLayer()

	a, b = functor(model)

	gs = gradient(m->mean(m(batch)), model)[1]
  #diff way to get the gradient, define the fuction inside do, don't pass the
	#use model parameters
	gs2=gradient(params(model)) do
		mean(model(batch))
	end

   #PROBLEM:  We still get nothing for the gradient of the embedding layer
	for p in params(model)
		println(typeof(gs2[p]))
	end
	model.pe.embedding
end


begin
	struct GPTPositionEmbedding{F, W <: AbstractArray{F}}
	    embedding::W
	end

	@functor GPTPositionEmbedding

	function GPTPositionEmbedding(size::Int, max_len::Int = 1024)
	    GPTPositionEmbedding( zeros(Float32,size, max_len))
	end


	function (pe::GPTPositionEmbedding{F})(x::AbstractArray{F}) where F
	    pe.embedding[:, 1:size(x,2)]  # just return the needed position embedding
	end


end


begin

   struct  GPTTestLayer
		embd::Embed
		pe::GPTPositionEmbedding
		linear::Positionwise
	end
		@functor GPTTestLayer

		function (t::GPTTestLayer)(x)
			we = t.embd(x, inv(sqrt(512)))
			e = we .+ t.pe(we)
			p = t.linear(e)
		end


	function GPTTestLayer()
		embd=Embed(512, length(vocab))
		pe = GPTPositionEmbedding(512, length(vocab))
		linear = Positionwise(Dense(512, length(vocab), initW=ones,initb=ones))
		GPTTestLayer(embd, pe, linear)
	end
	model=GPTTestLayer()

	for i in 1:10
		ps = params(model)

		gs2=gradient(ps) do
			mean(model(batch))
		end

		for p in ps
			println(size(p), mean(p), mean(gs2[p]), typeof(gs2[p]), size(gs2[p]))
		end
		opt = ADAM(1e-4)
		update!(opt, ps, gs2)
	end
end



#Try simple model with straight array as possition embedding.
begin


	mutable struct  simpleLayerArrayPosEmd
		embd::Embed
		pe
		#linear::Positionwise
	end
		@functor simpleLayerArrayPosEmd

		function (t::simpleLayerArrayPosEmd)(x)
			we = t.embd(x, inv(sqrt(512)))
			@show size(we)

			e = we .+ t.pe[:, 1:size(we,2)]
			#p = t.linear(e)
		end


	function simpleLayerArrayPosEmd()
		embd=Embed(512, length(vocab))
		pe = zeros(512,100)   ## pos embedding is larger that training, so some will remain zero
		#linear = Positionwise(Dense(512, length(vocab)))
		simpleLayerArrayPosEmd(embd, pe) #, linear)
	end
	model=simpleLayerArrayPosEmd()

    model(batch)
	ps = params(model)
	gs3=gradient(ps) do
		mean(model(batch))
	end

	for p in ps
		println(typeof(gs3[p]))
	end

	for i in 1:10

		gs4=gradient(ps) do
			mean(model(batch))
		end

		for p in ps
			println(size(p), mean(p), mean(gs4[p]), typeof(gs4[p]), size(gs4[p]))
		end
		opt = ADAM(1e-4)
		update!(opt, ps, gs4)
	end

end
