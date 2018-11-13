<template>
    <div id="projects">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/">MorphoCluster</router-link>
            <ul class="navbar-nav mr-auto">
                <li class="nav-item active text-light">
                    Projects
                </li>
            </ul>
        </nav>
        <div class="scrollable">
            <div class="container">
                <b-table id="projects_table" striped sort-by="name" :items="projects" :fields="fields" showEmpty>
                    <template slot="table-colgroup">
                        <col class="col-wide" />
                        <col class="col-narrow" />
                    </template>
                    <template slot="name" slot-scope="data">
                        <router-link :to="{name: 'project', params: {project_id: data.item.project_id}}">{{data.item.name}}</router-link>
                    </template>
                    <div slot="progress" slot-scope="data">
                        <b-progress v-if="'progress' in data.item" :max="data.item.progress.leaves_n_nodes">
                            <b-progress-bar variant="success" :value="data.item.progress.leaves_n_filled_nodes" v-b-tooltip.hover :title="`${data.item.progress.leaves_n_filled_nodes} filled up`" />
                            <b-progress-bar variant="warning" :value="data.item.progress.leaves_n_approved_nodes - data.item.progress.leaves_n_filled_nodes" v-b-tooltip.hover :title="`${data.item.progress.leaves_n_approved_nodes} approved`" />
                        </b-progress>
                    </div>
                    <template slot="action" slot-scope="data">
                        <b-button size="sm" variant="primary" class="mr-2" :to="{name: 'approve', params: {project_id: data.item.project_id}}">
                            Approve
                        </b-button>
                        <b-button size="sm" variant="primary" class="mr-2" :to="{name: 'bisect', params: {project_id: data.item.project_id}}">Bisect</b-button>
                        <b-button size="sm" variant="primary" class="mr-2" @click.prevent="save_project(data.item.project_id)">Save</b-button>
                    </template>
                    <template slot="visible" slot-scope="data">
                        {{data.visible ? "yes" : "no"}}
                    </template>
                    <template slot="empty">
                        <div class="text-center">No projects available.</div>
                    </template>
                </b-table>
                <p v-if="!projects">No projects available.</p>
                <div style="margin: 0 auto; width: 0;">
                    <b-button size="sm" variant="danger" class="mr-2" href="/labeling">
                        Expert mode
                    </b-button>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";

export default {
    name: "projects",
    props: {},
    components: {},
    data() {
        return {
            fields: [
                //{ key: "project_id", sortable: true },
                { key: "name", sortable: true },
                "progress",
                "action"
            ],
            projects: []
        };
    },
    methods: {
        save_project(project_id) {
            console.log("Saving", project_id, "...");
            api.saveProject(project_id).then(result => {
                alert("Project saved: " + result["tree_fn"]);
            });
        }
    },
    mounted() {
        // Load node info
        api
            .getProjects()
            .then(projects => {
                this.projects = projects;

                this.projects.forEach(p => {
                    api.getNodeProgress(p.node_id).then(progress => {
                        console.log(`Got progress for ${p.node_id}.`);
                        this.$set(p, "progress", progress);
                    });
                });
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
#projects {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#projects_table tr td:nth-child(1) {
    width: 100%;
}

#projects_table tr td:not(:nth-child(1)) {
    width: auto;
    text-align: right;
    white-space: nowrap;
}

.scrollable {
    overflow-y: auto;
}
</style>
