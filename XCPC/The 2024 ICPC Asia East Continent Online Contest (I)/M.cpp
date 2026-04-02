#include <bits/stdc++.h>
using namespace std;

int vis[100005][26];
void solve()
{
    int n;
    cin >> n;
    for(int i=0;i<=n;i++)
    {
        for(int j=0;j<26;j++)
        {
            vis[i][j] = 0;
        }
    }
    //set<string> ss;
    map<string, int> mp;
    int cnt = 0;
    vector<int> a(28, 0);
    while (n--)
    {
        string s, t;
        char c;
        cin >> s >> c >> t;
        if (t == "accepted")
        {
            if (mp.count(s) == 0)
            {
                mp[s] = cnt;
                cnt++;
            }
            int id = mp[s];
            if(!vis[id][c - 'A'])
            {
                a[c - 'A']++;
                vis[id][c - 'A'] = 1;
            }
        }
    }
    int ans = -1;
    int id = -1;
    for (int i = 0; i < 26; i++)
    {
        if (a[i] > ans)
        {
            ans = a[i];
            id = i;
        }
    }
    cout << char('A' + id) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
