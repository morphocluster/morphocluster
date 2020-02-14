import Vue from 'vue';

export default new Vue({
    data: () => {
        return { breadcrumbs: [], loading: [] }
    },
    methods: {
        setBreadcrumbs(breadcrumbs) {
            console.log("setBreadcrumbs", breadcrumbs);
            this.breadcrumbs = breadcrumbs;
        },
        setLoading(name) {
            if (Vue.config.debug) {
                console.log(`Loading ${name}...`);
            }
            this.loading.push(name);
        },
        unsetLoading(name) {
            if (Vue.config.debug) {
                console.log(`Done loading ${name}...`);
            }
            this.loading = this.loading.filter(item => item !== name);
        }
    }
})