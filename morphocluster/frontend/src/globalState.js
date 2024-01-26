import Vue from 'vue';

export default new Vue({
    data: () => {
        return { breadcrumbs: [], loading: [] }
    },
    methods: {
        /**
         * Set the breadcrumbs in the app bar.
         * @param breadcrumbs - A list of router location objects which will be rendered as breadcrumbs.
         * @see https://v3.router.vuejs.org/guide/essentials/navigation.html
         * @see https://v2.vuetifyjs.com/en/components/breadcrumbs/
         * @example
         *      setBreadcrumbs([{name: "foo", text: "Foo"}, {name: "foo", params: {route_parameter: "bar"}, text: "Bar"}])
         */
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