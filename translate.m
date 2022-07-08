discardProp = 0.30;

downloadFolder = tempdir;
url = "http://www.manythings.org/anki/cmn-eng.zip";
filename = fullfile(downloadFolder,"cmn-eng.zip");
dataFolder = fullfile(downloadFolder,"cmn-eng");

if ~exist(dataFolder,"dir")
    fprintf("Downloading English-Chinese Tab-delimited Bilingual Sentence Pairs data set (7.6 MB)... ")
    websave(filename,url);
    unzip(filename,dataFolder);
    fprintf("Done.\n")
end


filename = fullfile(dataFolder,"cmn.txt");

opts = delimitedTextImportOptions(...
    Delimiter="\t", ...
    VariableNames=["Source" "Target" "License"], ...
    SelectedVariableNames=["Source" "Target"], ...
    VariableTypes=["string" "string" "string"], ...
    Encoding="UTF-8");

data = readtable(filename, opts);
head(data)


idx = size(data,1) - floor(discardProp*size(data,1)) + 1;
data(idx:end,:) = [];


size(data,1)


trainingProp = 0.9;
idx = randperm(size(data,1),floor(trainingProp*size(data,1)));
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];



head(dataTrain)



numObservationsTrain = size(dataTrain,1)


documentsGerman = preprocessText(dataTrain.Source);

encGerman = wordEncoding(documentsGerman);

documentsEnglish = preprocessText(dataTrain.Target);


encEnglish = wordEncoding(documentsEnglish);
numWordsGerman = encGerman.NumWords
numWordsEnglish = encEnglish.NumWords


embeddingDimension = 128;
numHiddenUnits = 128;

[lgraphEncoder,lgraphDecoder] = languageTranslationLayers(embeddingDimension,numHiddenUnits,numWordsGerman,numWordsEnglish);

netEncoder = dlnetwork(lgraphEncoder);
netDecoder = dlnetwork(lgraphDecoder);


netDecoder.OutputNames = ["softmax" "context" "lstm2/hidden" "lstm2/cell"];

miniBatchSize = 64;
numEpochs = 20;
learnRate = 0.005;

gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

epsilonStart = 0.5;
epsilonEnd = 0;

sequenceLengths = doclength(documentsGerman);
[~,idx] = sort(sequenceLengths);
documentsGerman = documentsGerman(idx);
documentsEnglish = documentsEnglish(idx);

adsSource = arrayDatastore(documentsGerman);
adsTarget = arrayDatastore(documentsEnglish);
cds = combine(adsSource,adsTarget);

mbq = minibatchqueue(cds,4, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,Y) preprocessMiniBatch(X,Y,encGerman,encEnglish), ...
    MiniBatchFormat=["CTB" "CTB" "CTB" "CTB"], ...
    PartialMiniBatch="discard");

figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));

xlabel("Iteration")
ylabel("Loss")
ylim([0 inf])
grid on

trailingAvgEncoder = [];
trailingAvgSqEncoder = [];
trailingAvgDecder = [];
trailingAvgSqDecoder = [];

numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
numIterations = numIterationsPerEpoch * numEpochs;
epsilon = linspace(epsilonStart,epsilonEnd,numIterations);

iteration = 0;
start = tic;
lossMin = inf;
reset(mbq)

% Loop over epochs.
for epoch = 1:numEpochs

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T,maskT,decoderInput] = next(mbq);
        
        % Compute loss and gradients.
        [loss,gradientsEncoder,gradientsDecoder,YPred] = dlfeval(@modelLoss,netEncoder,netDecoder,X,T,maskT,decoderInput,epsilon(iteration));
        
        % Update network learnable parameters using adamupdate.
        [netEncoder, trailingAvgEncoder, trailingAvgSqEncoder] = adamupdate(netEncoder,gradientsEncoder,trailingAvgEncoder,trailingAvgSqEncoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        [netDecoder, trailingAvgDecder, trailingAvgSqDecoder] = adamupdate(netDecoder,gradientsDecoder,trailingAvgDecder,trailingAvgSqDecoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        % Generate translation for plot.
        if iteration == 1 || mod(iteration,10) == 0
            strGerman = ind2str(X(:,1,:),encGerman);
            strEnglish = ind2str(T(:,1,:),encEnglish,Mask=maskT);
            strTranslated = ind2str(YPred(:,1,:),encEnglish);
        end

        % Display training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(gather(extractdata(loss)));
        addpoints(lineLossTrain,iteration,loss)
        title( ...
            "Epoch: " + epoch + ", Elapsed: " + string(D) + newline + ...
            "Source: " + strGerman + newline + ...
            "Target: " + strEnglish + newline + ...
            "Training Translation: " + strTranslated)

        drawnow
        
        % Save best network.
        if loss < lossMin
            lossMin = loss;
            netBest.netEncoder = netEncoder;
            netBest.netDecoder = netDecoder;
            netBest.loss = loss;
            netBest.iteration = iteration;
            netBest.D = D;
        end
    end

    % Shuffle.
    shuffle(mbq);
end

netBest.encGerman = encGerman;
netBest.encEnglish = encEnglish;

D = datetime("now",Format="yyyy_MM_dd__HH_mm_ss");
filename = "net_best__" + string(D) + ".mat";
save(filename,"netBest");

netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;

