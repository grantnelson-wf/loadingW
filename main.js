require.config({
    paths: {
        src: 'src'
    }
});

require(['src/loadingW'], function(LoadingW) {

    var loadingW = new LoadingW();
    loadingW.setup('loadingWTarget');
    loadingW.show();
    //loadingW.hide();
        
});
