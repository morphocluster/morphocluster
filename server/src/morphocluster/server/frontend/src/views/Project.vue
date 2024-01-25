<template>
    <div id="project">
        <v-container>
            <v-card>
                <v-card-title>Projektinformationen</v-card-title>
                <v-card-text>
                    <v-row v-for="(value, key) in project" :key="key">
                        <v-col>{{ key }}</v-col>
                        <v-col>{{ value }}</v-col>
                    </v-row>
                </v-card-text>
            </v-card>
            <v-row justify="center" class="my-2">

                <v-btn color="primary" @click.prevent="showSaveModal">Save Project</v-btn>
            </v-row>
        </v-container>

        <v-dialog v-model="saveModalVisible" max-width="500">
            <v-card>
                <v-card-title>{{ saveTitle }}</v-card-title>
                <v-card-text>
                    <v-form @submit.prevent="handleSaveOk">
                        <v-container>
                            <v-row>
                                <v-col cols="12">
                                    <v-text-field v-model="saveSlug" label="Name"
                                        placeholder="Enter a name for the saved file"></v-text-field>
                                </v-col>
                            </v-row>
                            <v-row v-if="saveSaving">
                                <v-col cols="12">Saving...</v-col>
                            </v-row>
                        </v-container>
                    </v-form>
                </v-card-text>
                <v-card-actions>
                    <v-btn @click="handleSaveOk">OK</v-btn>
                    <v-btn @click="saveModalVisible = false">Cancel</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </div>
</template>
  
<script>
import * as api from "@/helpers/api.js";
import axios from "axios";
import DarkModeControl from "@/components/DarkModeControl.vue";
import state from "../globalState.js";

export default {
    name: "ProjectView",
    props: { "project_id": Number },
    components: { DarkModeControl },
    data() {
        return {
            project: null,
            saveSlug: "",
            saveTitle: "",
            saveProjectId: null,
            saveSaving: false,
            saveModalVisible: false,
        };
    },
    methods: {
        async initialize() {
            state.setBreadcrumbs(["project", this.project_id], "project");
        },
        showSaveModal() {
            this.saveSlug = this.project.name;
            this.saveProjectId = this.project.project_id;
            this.saveTitle = `Save ${this.project.name} (${this.saveProjectId})`;
            this.saveModalVisible = true;
        },
        handleSaveOk(evt) {
            evt.preventDefault();

            console.log("Saving", this.saveProjectId, "...");
            this.saveSaving = true;
            api.saveProject(this.saveProjectId)
                .then((result) => {
                    console.log("Project saved: " + result.url);
                    this.saveSaving = false;
                    this.$nextTick(() => {
                        this.saveModalVisible = false;
                    });
                    this.$nextTick(() => {
                        window.open(result.url + "?download=1");
                    });
                })
        },
    },
    mounted() {
        // Load node info
        this.initialize();
        axios
            .get(`/api/projects/${this.project_id}`)
            .then((response) => {
                this.project = response.data;
                console.log(response.data);
            })
            .catch((e) => {
                console.log(e);
            });
    },
};
</script>
  
<style>
#project {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
</style>
  