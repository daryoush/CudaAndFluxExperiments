##

using Flux
using CUDA
using Transformers
using Transformers.Basic
using Flux: onehot, onecold
using Transformers: PositionEmbedding
#flux function for update parameters
using Flux: gradient,  @functor
using Flux.Optimise: update!
using Flux:onehotbatch, logitcrossentropy
using Statistics: norm, mean, std
using Flux: Momentum
## ---  Model and Datasoruce


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

######
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
#
# sample = preprocess.(sample_data())
# encoded_sample = vocab(sample[1]) #use Vocabulary to encode the training data
#
struct  SimpleTransfomerEncoder
	embd::Embed
	pe::GPTPositionEmbedding
	#pe::PositionEmbedding
	encode1::Transformer
	encode2::Transformer
end

@functor SimpleTransfomerEncoder

struct  SimpleTransfomerDecoder
	embd::Embed
	pe::GPTPositionEmbedding
	#pe::PositionEmbedding
	decode1::TransformerDecoder
	decode2::TransformerDecoder
	linear::Positionwise
end

@functor SimpleTransfomerDecoder


function SimpleTransfomer()
	#512 is the size of internal embedding representation
	embd=Embed(512, length(vocab))
	pe = GPTPositionEmbedding(512)
	#pe = PositionEmbedding(512, maxInputLen; trainable=true)   # max length of input string

	embd_dec=Embed(512, length(vocab))
	pe_dec = GPTPositionEmbedding(512)
	#pe_dec = PositionEmbedding(512, maxInputLen; trainable=true)

	encode_t1 = Transformer(512, 8, 64, 2048)
	encode_t2 = Transformer(512, 8, 64, 2048)
	decode_t1 = TransformerDecoder(512, 8, 64, 2048)
	decode_t2 = TransformerDecoder(512, 8, 64, 2048)
	linear = Positionwise(Dense(512, length(vocab)))
    encoder=SimpleTransfomerEncoder(embd, pe, encode_t1, encode_t2)
	decoder=SimpleTransfomerDecoder(embd_dec, pe_dec, decode_t1,decode_t2, linear)
	encoder, decoder
end

function (t::SimpleTransfomerEncoder)(x)
	we = t.embd(x, inv(sqrt(512)))
	e = we .+ t.pe(we)
	e=t.encode1(e)
	t.encode2(e)
end
function (t::SimpleTransfomerDecoder)(x, m)
	we = t.embd(x, inv(sqrt(512)))
	e = we .+ t.pe(we)
	e = t.decode1(e, m)
	e = t.decode2(e, m)
	t.linear(e)
end

## ---

opt = ADAM(1e-4)
#opt=Momentum(0.001)
model = SimpleTransfomer()

## ---

using Flux: onecold
# assume x is list of numbers, This method will add start and end symbol,
# encode it as vocab, then feed it to the network and decode the output
# resulting output should have begin and end symbol
#list of numbers to a list of numbers with start and finish.
function translate(x, encoder, decoder)
    seq = [startsym]
	ix=vocab.(preprocess(x))  #add start and end, and vocab encoded
	encoderBatchOutput=encoder(ix)

	len = length(ix)
	local modelOut
    for i = 1:2len     # go twice as long, stop when there is end symbol
		modelOut = decoder(vocab(seq), encoderBatchOutput)
		dec = softmax(modelOut)  # distribution over vocab
		ntok = onecold(vocab, dec)  # decode from vocab to labels
        push!(seq, ntok[end])
        if ntok[end] == endsym
		 	break
		 end
    end
	seq   # note seq contains decoded output from itereation of the loop, one symbol at the time
  end

## ---

function train!(enc, dec, cnt=2000)
	@info "start training"
	local l
	local y, yhat
	ps=params(enc, dec)
	for i = 0:cnt
		trainingSize = 1:100
		batch=hcat((vocab.(preprocess(sample_data())) for i = trainingSize)...)
		#view the data as one long stream, then onehot encode it

		#assume output should be like input
		expected=batch[2:end,:]
		expectedStream=onehot(vocab, view(expected, :))  # using vocab one hot the array

		gs = gradient(ps) do
	        input = batch[1:end-1, :]
			encoderBatchOutput= enc(input)
			yhat = dec(input, encoderBatchOutput)   # output should be vocabs
			generatedStream = reshape(yhat, size(yhat)[1],:)

			l=logitcrossentropy(generatedStream, expectedStream)
		end
		if i % 10 == 0
			validateModel( enc, dec)
			@show i, l
		end
		update!(opt, ps, gs)
	end
end

## ---

function lossMeasure(b, yhat)
	expected = view(b, :)  # one dim list of output, vocab encoded
    y=onehot(vocab, expected)
	generatedStream = reshape(yhat, size(yhat)[1],:)
	logitcrossentropy(generatedStream, y)
end

# assuming batch is vocab encoded,
function doAModelRun(batch, encoder, decoder)
	encoderBatchOutput=encoder(batch)
	yhat = decoder(batch, encoderBatchOutput)
	@show loss=lossMeasure(batch, yhat)
end

function validateModel(encoder, decoder)
    validationData = [5,5,6,4,1,2,3,4,7,10]
	validationResult = translate(validationData, encoder, decoder)
	println(validationData, "->", validationResult)
end

## ---

train!(model...)
## ---  In case we need to do a step by step training

before = validateModel(model...)
@show before
#Get a batch, try run the model for a batch before and after a single training loop
batch=hcat(vocab.(preprocess(sample_data()) for i = 1:2)...)
@show doAModelRun(batch, model...)
train!(model..., 1)
@show doAModelRun(batch, model...)

#Translate the fix data again
after = validateModel(model ...)
@show after
