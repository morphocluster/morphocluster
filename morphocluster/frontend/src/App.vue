<template>
    <v-app>
        <v-app-bar app color="grey darken-3" dark>
            <div class="d-flex align-center">
                <router-link to="/">
                    <v-img alt="MorphoCluster Logo" class="shrink mr-2" contain src="@/assets/morphocluster.png"
                        transition="scale-transition" width="40" />
                </router-link>
            </div>

            <v-toolbar-title>
                <router-link to="/">MorphoCluster</router-link>
            </v-toolbar-title>

            <v-breadcrumbs :items="globalState.breadcrumbs" large>
                <template v-slot:item="{ item }">
                    <v-breadcrumbs-item :to="{ name: item.name, params: item.params }" exact>
                        {{ item.text }}
                    </v-breadcrumbs-item>
                </template>
            </v-breadcrumbs>
            <v-spacer></v-spacer>
            <dark-mode-control />
        </v-app-bar>
        <v-main>
            <v-tooltip top>
                <template v-slot:activator="{ on }">
                    <v-progress-linear v-on="on" :active="globalState.loading.length > 0" absolute indeterminate />
                </template>
                <span>Loading {{ globalState.loading.join(", ") }}...</span>
            </v-tooltip>
            <router-view />
        </v-main>
        <!-- <v-footer app>Footer</v-footer> -->
    </v-app>
</template>

<script>
import globalState from "@/globalState.js";
import DarkModeControl from "@/components/DarkModeControl.vue";


export default {
    name: "MorphoCluster",
    components: { DarkModeControl },
    data: () => {
        return { globalState };
    }
};
</script>
<style>
.v-toolbar__title a {
    color: white !important;
    text-decoration: none;
}
</style>