<template>
    <div id="projects">
        <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
            <router-link class="navbar-brand" :to="{ name: 'home' }">MorphoCluster</router-link>
            <ul class="navbar-nav nav-item">
                <li class="nav-item">
                    <router-link class="nav-link" :to="{ name: 'projects' }">Projects</router-link>
                </li>
            </ul>
            <dark-mode-control class="ml-auto" />
        </nav>
        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table id="projects_table" striped sort-by="name" :items="projects" :fields="fields" showEmpty>
                    <template slot="table-colgroup">
                        <col class="col-wide" />
                        <col class="col-narrow" />
                    </template>
                    <template v-slot:cell(name)="data">
                        <router-link :to="{
                            name: 'project',
                            params: { project_id: data.item.project_id },
                        }">{{ data.item.name }}</router-link>
                    </template>
                    <template v-slot:cell(progress)="data">
                        <!-- validated / grown clusters -->
                        <b-progress v-if="'progress' in data.item" :max="data.item.progress.leaves_n_nodes" class="mb-1">
                            <b-progress-bar variant="success" :value="data.item.progress.leaves_n_filled_nodes
                            " v-b-tooltip.hover
                                :title="`${data.item.progress.leaves_n_filled_nodes} / ${data.item.progress.leaves_n_nodes} clusters grown`" />
                            <b-progress-bar variant="warning" :value="data.item.progress.leaves_n_approved_nodes -
                                data.item.progress.leaves_n_filled_nodes
                            " v-b-tooltip.hover :title="`${Humanize.compactInteger(
    data.item.progress.leaves_n_approved_nodes,
    1
)} / ${Humanize.compactInteger(
    data.item.progress.leaves_n_nodes,
    1
)} clusters validated`" />
                        </b-progress>
                        <!-- objects in clusters -->
                        <b-progress v-if="'progress' in data.item" :max="data.item.progress.n_objects_deep" class="mb-1"
                            :title="`${Humanize.compactInteger(
                                data.item.progress.leaves_n_approved_objects,
                                1
                            )} / ${Humanize.compactInteger(
                                data.item.progress.n_objects_deep,
                                1
                            )} (${Math.round(
                                (data.item.progress.leaves_n_approved_objects /
                                    data.item.progress.n_objects_deep) *
                                100
                            )}%) objects in validated clusters`">
                            <b-progress-bar variant="success" :value="data.item.progress.leaves_n_approved_objects
                            " v-b-tooltip.hover />
                        </b-progress>
                    </template>
                    <template v-slot:cell(action)="data">
                        <b-button size="sm" variant="primary" class="mr-2" :to="{
                            name: 'approve',
                            params: { project_id: data.item.project_id },
                        }">
                            Validate
                        </b-button>
                        <b-button size="sm" variant="primary" class="mr-2" :to="{
                            name: 'bisect',
                            params: { project_id: data.item.project_id },
                        }">Grow</b-button>
                    </template>
                    <template v-slot:empty>
                        <div class="text-center">No projects available.</div>
                    </template>
                </b-table>
                <p v-if="!projects">No projects available.</p>
                <div style="margin: 0 auto; width: 0">
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
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import DarkModeControl from "@/components/DarkModeControl.vue";


import Humanize from "humanize-plus";

export default {
    name: "ProjectsView",
    props: {},
    components: { DarkModeControl },
    data() {
        return {
            fields: [
                //{ key: "project_id", sortable: true },
                { key: "name", sortable: true },
                "progress",
                "action",
            ],
            projects: [],
            alerts: [],
            // Make Humanize available in template:
            Humanize,
        };
    },
    methods: {

    },
    mounted() {
        // Load node info
        api.getProjects()
            .then((projects) => {
                this.projects = projects;

                return Promise.all(
                    this.projects.map((p) => {
                        return api
                            .getNodeProgress(p.node_id)
                            .then((progress) => {
                                console.log(`Got progress for ${p.node_id}.`);
                                this.$set(p, "progress", progress);
                            });
                    })
                );
            })
            .catch((e) => {
                console.log(e);
                // TODO: Use axiosErrorHandler
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger",
                });
            });
    },
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

.alerts {
    padding-top: 1em;
}
</style>
