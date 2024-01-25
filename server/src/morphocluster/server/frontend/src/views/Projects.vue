<template>
    <div id="projects">
        <div class="container">
            <div class="alerts" v-if="alerts.length">
                <v-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                    {{ a.message }}
                </v-alert>
            </div>
            <v-data-table sort-by="name" :items="projects" :headers="headers" showEmpty hide-default-footer>
                <template v-slot:item.name="{ item }">
                    <router-link :to="{
                        name: 'project',
                        params: { project_id: item.project_id },
                    }">{{ item.name }}</router-link>
                </template>

                <template v-slot:item.progress="data">
                    <!-- validated / grown clusters -->
                    <div style="border: 2px solid #717171; border-radius: 6px; margin-bottom: 5px; ">
                        <v-progress-linear rounded variant="success"
                            :value="data.item.progress.leaves_n_filled_nodes / data.item.progress.leaves_n_nodes * 100"
                            :buffer-value="data.item.progress.leaves_n_approved_nodes / data.item.progress.leaves_n_nodes * 100"
                            background-color="yellow" color="green" height="7" style="margin: 0;" :title="`${data.item.progress.leaves_n_filled_nodes} / ${data.item.progress.leaves_n_nodes} clusters grown, ${Humanize.compactInteger(
                                data.item.progress.leaves_n_approved_nodes,
                                1
                            )} / ${Humanize.compactInteger(
                                data.item.progress.leaves_n_nodes,
                                1
                            )} clusters validated`" />
                    </div>

                    <div style="border: 2px solid #717171; border-radius: 6px; margin-bottom: 5px; ">
                        <v-progress-linear rounded variant="success"
                            :value="data.item.progress.leaves_n_approved_objects / data.item.progress.n_objects_deep * 100"
                            background-color="white" color="green" height="7" style="margin: 0;" :title="`${Humanize.compactInteger(
                                data.item.progress.leaves_n_approved_objects,
                                1
                            )} / ${Humanize.compactInteger(
                                data.item.progress.n_objects_deep,
                                1
                            )} (${Math.round(
                                (data.item.progress.leaves_n_approved_objects /
                                    data.item.progress.n_objects_deep) *
                                100
                            )}%) objects in validated clusters`" />
                    </div>
                </template>
                <template v-slot:item.action="{ item }">
                    <v-btn small variant="primary" class="mr-2" :to="{
                        name: 'approve',
                        params: { project_id: item.project_id },
                    }">
                        Validate
                    </v-btn>
                    <v-btn small variant="primary" class="mr-2" :to="{
                        name: 'bisect',
                        params: { project_id: item.project_id },
                    }">Grow</v-btn>
                </template>

                <template v-slot:empty>
                    <div class="text-center">No projects available.</div>
                </template>
            </v-data-table>
            <p v-if="!projects">No projects available.</p>
            <div style="margin: 0 auto; width: 0">
                <v-btn large color="primary" class="mr-2">
                    Expert mode
                </v-btn>
            </div>

        </div>
    </div>
</template>

<script>
import globalState from "@/globalState.js";
import * as api from "@/helpers/api.js";
import state from "../globalState.js";
import Humanize from "humanize-plus";

export default {
    name: "ProjectsView",
    props: {},
    data() {
        return {
            headers: [
                { text: "Name", value: "name", sortable: true },
                { text: "Progress", value: "progress" },
                { text: "Action", value: "action" },
            ],
            projects: [],
            alerts: [],
            Humanize,
        };
    },
    methods: {
        async initialize() {
            state.setBreadcrumbs(["projects"], "projects");
        }
    },
    mounted() {
        this.initialize();
        api.getProjects()
            .then((projects) => {
                this.projects = projects;
                console.log(projects[0].name);
                const getProgressPromises = this.projects.map((p) => {
                    return api.getNodeProgress(p.node_id)
                        .then((progress) => {
                            console.log(`Got progress for ${p.node_id}.`);
                            this.$set(p, "progress", progress);
                        })
                        .catch((progressError) => {
                            console.error(`Error getting progress for ${p.node_id}.`, progressError);
                        });
                });
                return Promise.all(getProgressPromises);
            })
            .catch((error) => {
                console.error(error);
                if (axiosErrorHandler) {
                    axiosErrorHandler(error);
                } else {
                    this.alerts.unshift({
                        message: "Error fetching projects.",
                        variant: "danger",
                    });
                }
            })
            .finally(() => {
                globalState.unsetLoading("Projects");
            });

    },
};
</script>

<style>
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
