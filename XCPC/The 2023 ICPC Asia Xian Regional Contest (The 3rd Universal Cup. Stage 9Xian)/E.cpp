#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 5005;
vector<int>mp[N+1];
bool bfs(int st)
{

}
void solve()
{
    int n;
    cin >> n;
    
    string s;
    for (int i = 1; i <= n; i++)
    {
        cin >> s;
        for (int j = 1; i <= n; j++)
        {
            if (s[j - 1] == '1')
            {
                mp[i].push_back(j);
            }
        }
    }
}
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int _t = 1;
    // cin>>_t;
    while (_t--)
    {
        solve();
    }
    return 0;
}