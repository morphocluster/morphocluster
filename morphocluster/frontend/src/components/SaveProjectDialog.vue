<template>
    <v-dialog
        v-if="project"
        v-model="saveDialogModal"
        max-width="600px"
        @click:outside="onCancel"
        @keydown.esc="onCancel"
    >
        <form @submit.prevent="handleSave">
            <v-card>
                <v-card-title class="headline">Save project {{project.name}}</v-card-title>
                <v-progress-linear :active="saving" indeterminate />
                <v-card-text>
                    <v-container>
                        <v-row>
                            <v-col cols="12">
                                <v-text-field
                                    :disabled="disableInput"
                                    label="Name*"
                                    :value="project.name"
                                    required
                                />
                            </v-col>
                        </v-row>
                    </v-container>
                </v-card-text>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn
                        color="secondary"
                        :disabled="disableInput"
                        @click.prevent="onCancel"
                    >Cancel</v-btn>
                    <v-btn color="success" :disabled="disableInput" @click.prevent="onSave">Save</v-btn>
                </v-card-actions>
            </v-card>
        </form>
    </v-dialog>
</template>
<script>
import * as api from "@/helpers/api.js";

export default {
    props: ["bus"],
    data: () => {
        return {
            project: null,
            disableInput: false,
            saving: false,
            saveDialogModal: false
        };
    },
    created() {
        this.bus.$on("showSaveProjectDialog", project => {
            this.project = project;
        });
    },
    watch: {
        project: function(project) {
            console.log("project", project);
            this.saveDialogModal = project !== null;
        }
    },
    methods: {
        onSave() {
            console.log("Saving", this.project.project_id, "...");
            this.saving = true;

            api.saveProject(this.project.project_id)
                .then(result => {
                    console.log("Project saved: " + result["tree_fn"]);
                    this.$emit("success", result);
                })
                .catch(e => {
                    this.$emit("error", e);
                })
                .finally(() => {
                    this.saving = false;
                    this.project = null;
                });
        },
        onCancel() {
            console.log("Cancel.");
            this.project = null;
        }
    }
};
</script>