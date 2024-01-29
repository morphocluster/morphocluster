<template>
    <div id="project" class="scrollable">
        <v-container>
            <h2>Project: {{ this.project.name }}</h2>
            <div style="display: flex; justify-content: center;">
                <table>
                    <tr v-for="(value, key) in project" :key="key">
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                </table>
            </div>


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
import mixins from "@/mixins.js";

export default {
    name: "ProjectView",
    mixins: [mixins],
    props: { "project_id": Number },
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
        axios
            .get(`/api/projects/${this.project_id}`)
            .then((response) => {
                console.log(response.data);
                this.project = response.data;
                this.setBreadcrumbs([{ name: 'projects', text: "Projects" }, { name: "project", text: this.project.name, params: { project_id: this.project.project_id } }]);
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


#project table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
}

#project td,
#project th {
    border: 1px solid #251919;
    text-align: left;
    padding: 8px;
}

#project tr:nth-child(even) {
    background-color: #dddddd;
}
</style>
  