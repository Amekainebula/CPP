#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n' 
using namespace std;
//int vis[1000005];
bool isperfectsquare(ull x)
{
    ull l=0,r=x;
    while(l<=r)
    {
        ull mid=(l+r)/2;
        if(mid*mid==x)
        {
            return true;
        }
        else if(mid*mid<x)
        {
            l=mid+1;
        }
        else
        {
            r=mid-1;
        }
    }
    return false;
}
void solve()
{
    ull n;
    cin >> n;
    ull sum=0;
    vector<int> ans;
    if(isperfectsquare((n*(n+1))/2))
    {
        cout<<"-1"<<endl;
        return;
    }
    else
    {
        for(ull i=1;i<=n;i++)
        {
            sum+=i;
            if(isperfectsquare(sum))
            {
                int cnt=0;
                while(isperfectsquare(sum))
                {
                    sum+=1;
                    cnt++;
                }
                ans.pb(i+cnt);
                for(ull j=i;j<i+cnt;j++)
                {
                    sum+=j;
                    if(isperfectsquare(sum))
                    {
                        sum+=1;
                        ans.pb(j+1);
                        ans.pb(j);
                        j++;
                        if(j>=i+cnt)
                        i++;
                    }
                    else
                    ans.pb(j);
                }
                i+=cnt;
            }
            else
            {
                ans.pb(i);
            }
        }
        for(ull i=0;i<n;i++)
        {
            cout<<ans[i]<<" ";
        }
        cout<<endl;
    }
    
    
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
   
    while (T--)
    {
        solve();
    }
    return 0;
}