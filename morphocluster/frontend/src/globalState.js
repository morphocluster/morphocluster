// globalState.js

import Vue from 'vue';

const state = new Vue({
    data: () => {
        return { breadcrumbs: [], loading: [] }
    },
    methods: {
        setBreadcrumbs(breadcrumbs, type) {
            console.log("setBreadcrumbs", breadcrumbs);
            let current_path = ""
            let itt = 0
            let toPath = ""
            const breadcrumbItems = breadcrumbs.map(breadcrumbText => {
                if (type === "files") {
                    if (itt === 0) {
                        toPath = { text: breadcrumbText, disabled: false, to: "/files/" };
                    } else if (itt === 1) {
                        toPath = { name: 'files', params: { file_path: breadcrumbText } }
                        current_path += breadcrumbText
                    } else {
                        current_path += "/" + breadcrumbText
                        toPath = { name: 'files', params: { file_path: current_path } }
                    }
                    itt += 1;
                } else if (type === "project" && itt === 1) {
                    toPath = { name: 'project', params: { project_id: breadcrumbText }, }
                    breadcrumbText = str(breadcrumbText)
                }
                else if (type === "projects" || (type === " project" && itt === 0)) {
                    toPath = { name: 'projects' }
                    itt += 1
                }
                return { text: breadcrumbText, disabled: false, to: toPath };
            });
            if (type == "home") {
                this.breadcrumbs = [];
            }
            this.breadcrumbs = breadcrumbItems;

            const titles = breadcrumbItems.map(b => b.text);

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
});

export default state;
export { state };
