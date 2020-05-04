import Vue from 'vue';

export default new Vue({
    data: () => {
        return { breadcrumbs: [], loading: [] }
    },
    methods: {
        setBreadcrumbs(breadcrumbs) {
            console.log("setBreadcrumbs", breadcrumbs);
            this.breadcrumbs = breadcrumbs;

            const titles = breadcrumbs.map(b => b.text);

            document.title = "MorphoCluster - " + titles.join(" / ");
        },
        setLoading(name) {
            if (Vue.config.debug) {
                console.log(`Loading ${name}...`);
            }
            this.loading.push(name);
        },
        unsetLoading(name) {
            if (Vue.config.debug) {
                console.log(`Done loading ${name}.`);
            }
            this.loading = this.loading.filter(item => item !== name);
        }
    }
})