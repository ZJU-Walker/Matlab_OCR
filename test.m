load('net_best__2022_06_18__23_43_50.mat');
encGerman = netBest.encGerman;
encEnglish = netBest.encEnglish;
netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;
strx = "hang on"
strGermanNew = [strx];
strTranslatedNew = translateText(netEncoder,netDecoder,encGerman,encEnglish,strGermanNew)