"use client";

import { useState } from "react";
import { format } from "date-fns";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { toast } from "@/components/ui/use-toast";
import { getErrorMessage } from "@/lib/error-message";
import { api } from "@/utils/api";
import { SEARCH_ENDPOINTS } from "@/utils/api-endpoints";
import { SearchMemory } from "@/types/api";

export default function PlaygroundPage() {
  const [userId, setUserId] = useState("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState("");
  const [threshold, setThreshold] = useState("");
  const [results, setResults] = useState<SearchMemory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedMemory, setSelectedMemory] = useState<SearchMemory | null>(
    null,
  );

  const handleSearch = async () => {
    const trimmedUserId = userId.trim();
    const trimmedQuery = query.trim();
    if (!trimmedUserId || !trimmedQuery) {
      toast({
        title: "Missing fields",
        description: "User ID and Query are required.",
        variant: "destructive",
      });
      return;
    }

    const body: {
      query: string;
      filters: { user_id: string };
      top_k?: number;
      threshold?: number;
    } = {
      query: trimmedQuery,
      filters: { user_id: trimmedUserId },
    };

    if (topK.trim()) {
      const parsed = Number(topK);
      if (!Number.isFinite(parsed) || parsed < 0) {
        toast({
          title: "Invalid top_k",
          description: "top_k must be a non-negative number.",
          variant: "destructive",
        });
        return;
      }
      body.top_k = parsed;
    }

    if (threshold.trim()) {
      const parsed = Number(threshold);
      if (!Number.isFinite(parsed)) {
        toast({
          title: "Invalid threshold",
          description: "threshold must be a number.",
          variant: "destructive",
        });
        return;
      }
      body.threshold = parsed;
    }

    setIsLoading(true);
    setHasSearched(true);
    try {
      const res = await api.post(SEARCH_ENDPOINTS.BASE, body);
      const raw = res.data?.results ?? res.data ?? [];
      setResults(Array.isArray(raw) ? raw : []);
    } catch (error) {
      setResults([]);
      toast({
        title: "Search failed",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const columns = [
    {
      key: "memory" as keyof SearchMemory,
      label: "Content",
      width: 400,
      render: (value: string) => (
        <span className="line-clamp-2 text-sm">{value}</span>
      ),
    },
    { key: "user_id" as keyof SearchMemory, label: "User", width: 100 },
    {
      key: "score" as keyof SearchMemory,
      label: "Score",
      width: 80,
      render: (value: number | undefined) =>
        typeof value === "number" ? value.toFixed(2) : "--",
    },
    {
      key: "created_at" as keyof SearchMemory,
      label: "Created",
      width: 120,
      render: (value: string) =>
        value ? format(new Date(value), "MMM d, yyyy") : "--",
    },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold font-fustat">Playground</h1>

      <Card className="border-memBorder-primary p-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="space-y-1.5">
            <Label htmlFor="playground-user-id">User ID</Label>
            <Input
              id="playground-user-id"
              placeholder="e.g. alice"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSearch();
              }}
            />
          </div>
          <div className="space-y-1.5 sm:col-span-2 lg:col-span-1">
            <Label htmlFor="playground-query">Query</Label>
            <Input
              id="playground-query"
              placeholder="Search query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSearch();
              }}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="playground-top-k">top_k</Label>
            <Input
              id="playground-top-k"
              type="number"
              min={0}
              placeholder="20"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSearch();
              }}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="playground-threshold">threshold</Label>
            <Input
              id="playground-threshold"
              type="number"
              step="0.01"
              min={0}
              max={1}
              placeholder="0.1"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleSearch();
              }}
            />
          </div>
        </div>
        <div className="mt-4">
          <Button onClick={() => void handleSearch()} disabled={isLoading}>
            <Search className="size-3.5 mr-1.5" />
            Search
          </Button>
        </div>
      </Card>

      {isLoading ? (
        <TableSkeleton rows={5} columns={4} />
      ) : !hasSearched ? (
        <EmptyState
          title="Search memories"
          description="Enter a User ID and query to run a semantic search against stored memories."
        />
      ) : results.length === 0 ? (
        <EmptyState
          title="No results"
          description="No memories matched this query for the given user."
        />
      ) : (
        <Card className="border-memBorder-primary overflow-hidden">
          <DataTable
            data={results}
            columns={columns}
            getRowKey={(row) => row.id}
            onRowClick={(row) => setSelectedMemory(row)}
            getRowClassName={(row) =>
              selectedMemory?.id === row.id
                ? "bg-surface-default-tertiary"
                : undefined
            }
          />
        </Card>
      )}

      <Sheet
        open={!!selectedMemory}
        onOpenChange={(open) => {
          if (!open) setSelectedMemory(null);
        }}
      >
        <SheetContent className="sm:max-w-md">
          <SheetHeader>
            <SheetTitle>Memory Detail</SheetTitle>
            <SheetDescription className="sr-only">
              View memory content and metadata
            </SheetDescription>
          </SheetHeader>
          {selectedMemory && (
            <div className="mt-6 space-y-4">
              <div className="space-y-1">
                <Label className="text-xs text-onSurface-default-tertiary">
                  Content
                </Label>
                <p className="text-sm">{selectedMemory.memory}</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label className="text-xs text-onSurface-default-tertiary">
                    ID
                  </Label>
                  <p className="text-xs font-mono break-all">
                    {selectedMemory.id}
                  </p>
                </div>
                {selectedMemory.user_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      User
                    </Label>
                    <p className="text-sm">{selectedMemory.user_id}</p>
                  </div>
                )}
                {typeof selectedMemory.score === "number" && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Score
                    </Label>
                    <p className="text-sm">{selectedMemory.score.toFixed(2)}</p>
                  </div>
                )}
                {selectedMemory.agent_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Agent
                    </Label>
                    <p className="text-sm">{selectedMemory.agent_id}</p>
                  </div>
                )}
                {selectedMemory.created_at && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Created
                    </Label>
                    <p className="text-sm">
                      {new Date(selectedMemory.created_at).toLocaleString()}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
}
