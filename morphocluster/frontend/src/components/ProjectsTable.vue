<template>
    <div class="projects-table">
        <v-data-table
            id="projects_table"
            disable-filtering
            disable-pagination
            hide-default-footer
            sort-by="name"
            :items="projects"
            :headers="headers"
        >
            <template v-slot:no-data>No projects.</template>
            <template v-slot:item.name="{ item }">
                <router-link
                    :to="{
                        name: 'project',
                        params: { project_id: item.project_id }
                    }"
                >{{ item.name }}</router-link>
            </template>
            <template v-slot:item.progress="{ item }">
                <v-tooltip top>
                    <template v-slot:activator="{ on }">
                        <v-progress-linear
                            rounded
                            color="warning"
                            :value="item.progress.leaves_n_approved_nodes"
                            v-on="on"
                            height="10"
                            class="my-1"
                        />
                    </template>
                    <span>{{item.progress.leaves_n_approved_nodes}} / {{item.progress.leaves_n_nodes}} validated</span>
                </v-tooltip>
                <v-tooltip top>
                    <template v-slot:activator="{ on }">
                        <v-progress-linear
                            rounded
                            color="success"
                            :value="item.progress.leaves_n_filled_nodes / item.progress.leaves_n_nodes * 100"
                            v-on="on"
                            height="10"
                            class="my-1"
                        />
                    </template>
                    <span>{{item.progress.leaves_n_filled_nodes}} / {{item.progress.leaves_n_nodes}} grown</span>
                </v-tooltip>
            </template>
            <template v-slot:item.action="{ item }">
                <v-btn
                    size="sm"
                    variant="primary"
                    class="mr-2"
                    :to="{
                        name: 'validate',
                        params: { project_id: item.project_id }
                    }"
                >Validate</v-btn>
                <v-btn
                    size="sm"
                    variant="primary"
                    class="mr-2"
                    :to="{
                        name: 'grow',
                        params: { project_id: item.project_id }
                    }"
                >Grow</v-btn>
                <v-btn size="sm" variant="primary" class="mr-2" @click.prevent="onSave(item)">Save</v-btn>
            </template>
            <template slot="visible" slot-scope="data">
                {{
                data.visible ? "yes" : "no"
                }}
            </template>
            <template slot="empty">
                <div class="text-center">No projects available.</div>
            </template>-->
        </v-data-table>
        <save-project-dialog :bus="bus" @success="onSaveSuccess" @error="onSaveError" />
        <!-- <b-modal ref="saveModal" lazy centered :title="save_title" @ok="HandleSaveOk">
            <div class="d-block text-center">
                <form @submit.stop.prevent="HandleSaveOk">
                    <b-container fluid>
                        <b-form-row class="my-1">
                            <b-col sm="3">
                                <label for="form_save_name">Name:</label>
                            </b-col>
                            <b-col sm="9">
                                <b-form-input
                                    type="text"
                                    placeholder="Enter a name for the saved file"
                                    id="form_save_name"
                                    v-model="save_slug"
                                ></b-form-input>
                            </b-col>
                        </b-form-row>
                        <b-row class="my-3" v-if="save_saving">
                            <b-col sm="12">Saving...</b-col>
                        </b-row>
                    </b-container>
                </form>
            </div>
        </b-modal>-->
    </div>
</template>

<script>
import Vue from "vue";

import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

// Components
import SaveProjectDialog from "@/components/SaveProjectDialog.vue";

export default {
    name: "projects-table",
    props: ["dataset_id"],
    components: { SaveProjectDialog },
    mixins: [mixins],
    data() {
        return {
            headers: [
                { text: "Project name", value: "name", sortable: true },
                {
                    text: "Progress",
                    value: "progress",
                    width: "200px",
                    sortable: false
                },
                { value: "action", sortable: false }
            ],
            projects: [],
            save_slug: "",
            save_title: "",
            save_project_id: null,
            save_saving: false,
            alerts: [],
            loading: true,

            // Project about to be saved
            save_project: null,

            // Local event bus
            bus: new Vue()
        };
    },
    methods: {
        onSave(project) {
            console.log("showSaveModal");
            this.bus.$emit("showSaveProjectDialog", project);
        },
        onSaveSuccess(result) {
            console.log(result);
        },
        onSaveError(evt) {
            console.log(evt);
        }
    },
    mounted() {
        // Load node info
        this.setLoading("projects");
        api.datasetGetProjects(this.dataset_id, true)
            .then(projects => {
                this.projects = projects;
            })
            .catch(e => {
                console.log(e);
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger"
                });
            })
            .finally(() => {
                this.loading = false;
                this.unsetLoading("projects");
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

.alerts {
    padding-top: 1em;
}
</style>
