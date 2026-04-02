// Code from Whalica
#include <bits/stdc++.h>

using i64 = long long;
using u64 = unsigned long long;

void solve()
{
}
int mx=INT32_MAX;
void brute()
{
    std::vector<int> a = {1, 2, 3, 4, 5, 6};
    int n = a.size();
    do
    {
        for (auto x : a)
            std::cout << x << " ";
        std::cout << "\n";
        std::set<std::vector<int>> s;
        auto dfs = [&](auto &&self, std::vector<int> f, int x, int sta, int l, int r) -> void
        {
            if (x >= n)
            {
                //            std::cerr << x << " " << sta << "\n";
                //            for (auto y : f) {
                //                std::cerr << y << " ";
                //            }
                //            std::cerr << "\n";
                s.insert(f);
                return;
            }
            else
            {
                if (sta == 0)
                {
                    if (r < l + 1)
                    {
                        s.insert(f);
                        return;
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        self(self, f, x + 1, i, l + 1, r);
                    }
                }
                else if (sta == 1)
                {
                    if (r - 1 < l)
                    {
                        s.insert(f);
                        return;
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        self(self, f, x + 1, i, l, r - 1);
                    }
                }
                else
                {
                    //                std::cerr << l << " " << r << "\n";
                    f.push_back(*std::min_element(a.begin() + l, a.begin() + r + 1));
                    for (int i = 0; i < 3; i++)
                    {
                        self(self, f, x + 1, i, l, r);
                    }
                }
            }
        };

        //    std::cerr << "1\n";

        for (int i = 0; i < 3; i++)
        {
            dfs(dfs, std::vector<int>(0), 0, i, 0, n - 1);
        }

        //    std::cerr << "2\n";

        std::cout << s.size() << "\n";
        mx=std::min(mx,(int)s.size());   
        for (auto x : s)
        {
            for (auto y : x)
            {
                std::cout << y << " ";
            }
            std::cout << "\n";
        }
        std::cout << "----------------------------\n";
    } while (next_permutation(a.begin(), a.end()));
    std::cout << mx << "\n";
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t = 1;
    //    std::cin >> t;

    while (t--)
    {
        //        solve();
        brute();
    }

    return 0;
}